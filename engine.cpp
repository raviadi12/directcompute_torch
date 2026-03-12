#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi.h>
#include <vector>
#include <unordered_map>
#include <string>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")

// GPU info for adaptive tuning
struct GPUInfo {
    uint32_t vendorId;          // 0x8086=Intel, 0x10DE=NVIDIA, 0x1002=AMD
    uint32_t deviceId;
    uint64_t dedicatedVRAM;     // bytes
    uint64_t sharedMemory;      // bytes
    uint32_t isIntegrated;      // 1=iGPU, 0=dGPU
    char description[128];
};
static GPUInfo g_gpuInfo = {};

ID3D11Device* g_device = nullptr;
ID3D11DeviceContext* g_context = nullptr;
std::unordered_map<std::string, ID3D11ComputeShader*> g_shaders;
ID3D11Buffer* g_cbCache = nullptr;
uint32_t g_cbCacheSize = 0;

// Static null arrays for unbinding — zeroed once at startup, reused every call
static ID3D11ShaderResourceView* g_nullSRVs[16] = {};
static ID3D11UnorderedAccessView* g_nullUAVs[16] = {};

// Track previously-bound slot counts for smart unbinding
static uint32_t g_prevSrvCount = 0;
static uint32_t g_prevUavCount = 0;

// Forward declaration: pool memory cap (set after GPU detection)
static uint64_t g_maxPoolMemBytes = 64 * 1024 * 1024;

static void FillGPUInfo(IDXGIAdapter* adapter) {
    DXGI_ADAPTER_DESC desc;
    if (FAILED(adapter->GetDesc(&desc))) return;
    g_gpuInfo.vendorId = desc.VendorId;
    g_gpuInfo.deviceId = desc.DeviceId;
    g_gpuInfo.dedicatedVRAM = desc.DedicatedVideoMemory;
    g_gpuInfo.sharedMemory = desc.SharedSystemMemory;
    for (int i = 0; i < 127 && desc.Description[i]; i++)
        g_gpuInfo.description[i] = (char)desc.Description[i];
    g_gpuInfo.description[127] = 0;
    // iGPU detection heuristic
    uint64_t vramMB = g_gpuInfo.dedicatedVRAM / (1024 * 1024);
    if (g_gpuInfo.vendorId == 0x8086 && vramMB < 512)
        g_gpuInfo.isIntegrated = 1;
    else if (vramMB < 256)
        g_gpuInfo.isIntegrated = 1;
    else
        g_gpuInfo.isIntegrated = 0;
}

extern "C" __declspec(dllexport) bool InitEngine() {
    if (g_device) return true;

    // ── Enumerate all adapters, pick the best discrete GPU ──
    IDXGIFactory* factory = nullptr;
    if (FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory)))
        return false;

    IDXGIAdapter* bestAdapter = nullptr;
    uint64_t bestVRAM = 0;
    bool bestIsDiscrete = false;

    IDXGIAdapter* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC desc;
        if (FAILED(adapter->GetDesc(&desc))) { adapter->Release(); continue; }

        // Skip Microsoft Basic Render Driver / software adapters
        if (desc.VendorId == 0x1414) { adapter->Release(); continue; }

        bool isDiscrete = true;
        uint64_t vramMB = desc.DedicatedVideoMemory / (1024 * 1024);
        if (desc.VendorId == 0x8086 && vramMB < 512) isDiscrete = false;
        else if (vramMB < 256) isDiscrete = false;

        // Prefer discrete over integrated, then most VRAM
        bool better = false;
        if (!bestAdapter) better = true;
        else if (isDiscrete && !bestIsDiscrete) better = true;
        else if (isDiscrete == bestIsDiscrete && desc.DedicatedVideoMemory > bestVRAM) better = true;

        if (better) {
            if (bestAdapter) bestAdapter->Release();
            bestAdapter = adapter;
            bestVRAM = desc.DedicatedVideoMemory;
            bestIsDiscrete = isDiscrete;
        } else {
            adapter->Release();
        }
    }

    if (!bestAdapter) { factory->Release(); return false; }

    // Create device on the chosen adapter (must use D3D_DRIVER_TYPE_UNKNOWN with explicit adapter)
    D3D_FEATURE_LEVEL fl;
    HRESULT hr = D3D11CreateDevice(bestAdapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0,
                                   nullptr, 0, D3D11_SDK_VERSION, &g_device, &fl, &g_context);
    if (FAILED(hr)) {
        bestAdapter->Release();
        factory->Release();
        return false;
    }

    FillGPUInfo(bestAdapter);
    bestAdapter->Release();
    factory->Release();
    
    // Set pool memory cap: iGPU uses shared memory, dGPU uses dedicated VRAM
    // 50% of usable memory, clamped [64MB, 512MB]
    uint64_t usableMem = g_gpuInfo.dedicatedVRAM;
    if (g_gpuInfo.isIntegrated && g_gpuInfo.sharedMemory > usableMem)
        usableMem = g_gpuInfo.sharedMemory;
    uint64_t halfMem = usableMem / 2;
    if (halfMem < 64 * 1024 * 1024) halfMem = 64 * 1024 * 1024;
    if (halfMem > 512 * 1024 * 1024) halfMem = 512 * 1024 * 1024;
    g_maxPoolMemBytes = halfMem;
    
    return true;
}

// Export GPU info struct for Python to read
extern "C" __declspec(dllexport) void GetGPUInfo(GPUInfo* out) {
    if (out) *out = g_gpuInfo;
}

extern "C" __declspec(dllexport) bool CompileShader(const char* name, const wchar_t* path) {
    if (!g_device) return false;
    ID3DBlob *blob = nullptr, *err = nullptr;
    HRESULT hr = D3DCompileFromFile(path, nullptr, nullptr, "CSMain", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &err);
    if (FAILED(hr)) {
        if (err) {
            OutputDebugStringA((char*)err->GetBufferPointer());
            err->Release();
        }
        return false;
    }
    ID3D11ComputeShader* s = nullptr;
    g_device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &s);
    g_shaders[name] = s;
    blob->Release();
    return true;
}

struct GPUBuffer {
    ID3D11Buffer* buffer;
    ID3D11ShaderResourceView* srv;
    ID3D11UnorderedAccessView* uav;
    uint32_t size;
    int refCount;
};

// --- BUFFER POOLS (unordered_map for O(1) lookup) ---
std::unordered_map<uint32_t, std::vector<GPUBuffer*>> g_bufferPool;
std::unordered_map<uint32_t, std::vector<ID3D11Buffer*>> g_stagingPool;

// Size-aware pool limits based on GPU's usable memory
// Old engine used flat max=8 for all sizes; we use similar generous limits.
static inline size_t MaxPoolForSize(uint32_t count) {
    size_t bytes = (size_t)count * sizeof(float);
    // iGPUs allocate from shared system memory — use that for pool sizing
    uint64_t usableMem = g_gpuInfo.dedicatedVRAM;
    if (g_gpuInfo.isIntegrated && g_gpuInfo.sharedMemory > usableMem)
        usableMem = g_gpuInfo.sharedMemory;
    if (usableMem > 0 && bytes >= usableMem / 4) return 4;  // truly huge: 4 copies
    if (bytes >= 4 * 1024 * 1024)    return 6;   // >= 4MB: generous
    if (bytes >= 256 * 1024)          return 8;   // >= 256KB: same as old
    return 16;                                     // < 256KB: aggressive
}
static uint64_t g_poolHits = 0;
static uint64_t g_poolMisses = 0;
static uint64_t g_poolMemBytes = 0;  // track total pool memory

extern "C" __declspec(dllexport) void* CreateBuffer(float* data, uint32_t count) {
    if (!g_device) return nullptr;
    GPUBuffer* b = nullptr;

    auto it = g_bufferPool.find(count);
    if (it != g_bufferPool.end() && !it->second.empty()) {
        b = it->second.back();
        it->second.pop_back();
        g_poolMemBytes -= (size_t)count * sizeof(float);
        b->refCount = 1;
        g_poolHits++;
        if (data) {
            g_context->UpdateSubresource(b->buffer, 0, nullptr, data, 0, 0);
        }
        return b;
    }
    g_poolMisses++;

    b = new GPUBuffer();
    b->size = count;
    b->refCount = 1;
    b->buffer = nullptr; b->srv = nullptr; b->uav = nullptr;

    D3D11_BUFFER_DESC desc = {};
    desc.ByteWidth = sizeof(float) * count;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = sizeof(float);

    D3D11_SUBRESOURCE_DATA sd = { data, 0, 0 };
    if (FAILED(g_device->CreateBuffer(&desc, data ? &sd : nullptr, &b->buffer))) {
        delete b; return nullptr;
    }

    // Structured SRV (for StructuredBuffer<float> in HLSL)
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    srvDesc.Buffer.FirstElement = 0;
    srvDesc.Buffer.NumElements = count;
    g_device->CreateShaderResourceView(b->buffer, &srvDesc, &b->srv);

    // Structured UAV (for RWStructuredBuffer<float> in HLSL)
    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_UNKNOWN;
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    uavDesc.Buffer.FirstElement = 0;
    uavDesc.Buffer.NumElements = count;
    uavDesc.Buffer.Flags = 0;
    g_device->CreateUnorderedAccessView(b->buffer, &uavDesc, &b->uav);
    return b;
}

extern "C" __declspec(dllexport) void AddRefBuffer(void* handle) {
    if (!handle) return;
    ((GPUBuffer*)handle)->refCount++;
}

// Upload new data to an existing buffer (avoids buffer creation overhead)
extern "C" __declspec(dllexport) void UpdateBuffer(void* handle, float* data) {
    if (!handle || !data || !g_context) return;
    GPUBuffer* b = (GPUBuffer*)handle;
    g_context->UpdateSubresource(b->buffer, 0, nullptr, data, 0, 0);
}

static inline void ReleaseBufferInternal(GPUBuffer* b) {
    auto& pool = g_bufferPool[b->size];
    size_t maxPool = MaxPoolForSize(b->size);
    size_t bufBytes = (size_t)b->size * sizeof(float);
    // Only enforce per-size limit (no global cap — matches old engine behavior).
    // D3D11 driver manages actual memory placement; pool just holds handles for reuse.
    if (pool.size() < maxPool) {
        pool.push_back(b);
        g_poolMemBytes += bufBytes;
    } else {
        if (b->srv) b->srv->Release();
        if (b->uav) b->uav->Release();
        if (b->buffer) b->buffer->Release();
        delete b;
    }
}

extern "C" __declspec(dllexport) void ReleaseBuffer(void* handle) {
    if (!handle) return;
    GPUBuffer* b = (GPUBuffer*)handle;
    if (--b->refCount == 0) {
        ReleaseBufferInternal(b);
    }
}

extern "C" __declspec(dllexport) void ReadBuffer(void* handle, float* dst) {
    if (!handle || !g_context || !g_device) return;
    g_context->Flush();
    GPUBuffer* b = (GPUBuffer*)handle;
    
    ID3D11Buffer* stage = nullptr;
    auto it = g_stagingPool.find(b->size);
    if (it != g_stagingPool.end() && !it->second.empty()) {
        stage = it->second.back();
        it->second.pop_back();
    } else {
        D3D11_BUFFER_DESC desc = { sizeof(float) * b->size, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ, 0, 0 };
        g_device->CreateBuffer(&desc, nullptr, &stage);
    }
    
    if (stage) {
        g_context->CopyResource(stage, b->buffer);
        D3D11_MAPPED_SUBRESOURCE map;
        if (SUCCEEDED(g_context->Map(stage, 0, D3D11_MAP_READ, 0, &map))) {
            memcpy(dst, map.pData, sizeof(float) * b->size);
            g_context->Unmap(stage, 0);
        }
        auto& pool = g_stagingPool[b->size];
        if (pool.size() < 2) {
            pool.push_back(stage);
        } else {
            stage->Release();
        }
    }
}

// Non-blocking flush: submit queued commands to GPU so it can start executing
// and free buffer memory from completed work. Does NOT wait for GPU completion.
extern "C" __declspec(dllexport) void FlushGPU() {
    if (g_context) g_context->Flush();
}

// Blocking GPU sync: submit commands AND wait until GPU finishes all work.
// Uses a D3D11 query event fence. Expensive — use only for timing/benchmarks.
extern "C" __declspec(dllexport) void SyncGPU() {
    if (!g_device || !g_context) return;
    
    D3D11_QUERY_DESC qd = {};
    qd.Query = D3D11_QUERY_EVENT;
    ID3D11Query* query = nullptr;
    if (FAILED(g_device->CreateQuery(&qd, &query))) return;
    
    g_context->End(query);
    g_context->Flush();
    
    BOOL done = FALSE;
    while (g_context->GetData(query, &done, sizeof(done), 0) == S_FALSE) {
        // Spin-wait for GPU completion
    }
    query->Release();
}

extern "C" __declspec(dllexport) void ClearBuffer(void* handle) {
    if (!handle || !g_context) return;
    GPUBuffer* b = (GPUBuffer*)handle;
    uint32_t clear[4] = {0,0,0,0};
    if (b->uav) g_context->ClearUnorderedAccessViewUint(b->uav, clear);
}

// Internal dispatch — takes shader pointer directly, no string lookup
static inline void DispatchInternal(ID3D11ComputeShader* s, void** srvs, uint32_t srvCount,
    void** uavs, uint32_t uavCount, uint32_t* threads, void* cb, uint32_t cbSize) {
    g_context->CSSetShader(s, nullptr, 0);
    
    ID3D11ShaderResourceView* srvPtrs[16] = {nullptr};
    for (uint32_t i = 0; i < srvCount && i < 16; ++i) srvPtrs[i] = srvs[i] ? ((GPUBuffer*)srvs[i])->srv : nullptr;
    g_context->CSSetShaderResources(0, srvCount, srvPtrs);
    
    ID3D11UnorderedAccessView* uavPtrs[16] = {nullptr};
    for (uint32_t i = 0; i < uavCount && i < 16; ++i) uavPtrs[i] = uavs[i] ? ((GPUBuffer*)uavs[i])->uav : nullptr;
    g_context->CSSetUnorderedAccessViews(0, uavCount, uavPtrs, nullptr);
    
    if (cb && cbSize > 0) {
        uint32_t alignedSize = (cbSize + 15) & ~15;
        if (!g_cbCache || g_cbCacheSize < alignedSize) {
            if (g_cbCache) g_cbCache->Release();
            D3D11_BUFFER_DESC cbd = { alignedSize, D3D11_USAGE_DYNAMIC, D3D11_BIND_CONSTANT_BUFFER, D3D11_CPU_ACCESS_WRITE, 0, 0 };
            g_device->CreateBuffer(&cbd, nullptr, &g_cbCache);
            g_cbCacheSize = alignedSize;
        }
        D3D11_MAPPED_SUBRESOURCE m;
        if (SUCCEEDED(g_context->Map(g_cbCache, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
            memcpy(m.pData, cb, cbSize);
            g_context->Unmap(g_cbCache, 0);
            g_context->CSSetConstantBuffers(0, 1, &g_cbCache);
        }
    }
    
    g_context->Dispatch(threads[0], threads[1], threads[2]);
    
    // Smart unbind: only clear slots actually used (max of current and previous)
    uint32_t clrSrv = srvCount > g_prevSrvCount ? srvCount : g_prevSrvCount;
    uint32_t clrUav = uavCount > g_prevUavCount ? uavCount : g_prevUavCount;
    if (clrSrv > 0) g_context->CSSetShaderResources(0, clrSrv, g_nullSRVs);
    if (clrUav > 0) g_context->CSSetUnorderedAccessViews(0, clrUav, g_nullUAVs, nullptr);
    g_prevSrvCount = srvCount;
    g_prevUavCount = uavCount;
}

extern "C" __declspec(dllexport) void RunShader(const char* name, void** srvs, uint32_t srvCount, void** uavs, uint32_t uavCount, uint32_t* threads, void* cb, uint32_t cbSize) {
    if (!g_context) return;
    auto it = g_shaders.find(name);
    if (it == g_shaders.end()) return;
    DispatchInternal(it->second, srvs, srvCount, uavs, uavCount, threads, cb, cbSize);
}

// Keep RunShaderRaw as alias to RunShader for Python compatibility
extern "C" __declspec(dllexport) void RunShaderRaw(const char* name, void** srvs, uint32_t srvCount, void** uavs, uint32_t uavCount, uint32_t* threads, void* cb, uint32_t cbSize) {
    RunShader(name, srvs, srvCount, uavs, uavCount, threads, cb, cbSize);
}

// ── Fused: CreateBuffer + RunShader in one call ──
extern "C" __declspec(dllexport) void* RunShaderAlloc(const char* name, void** srvs, uint32_t srvCount,
    uint32_t outCount, uint32_t* threads, void* cb, uint32_t cbSize) {
    void* outBuf = CreateBuffer(nullptr, outCount);
    if (!outBuf) return nullptr;
    void* uavs[1] = { outBuf };
    RunShader(name, srvs, srvCount, uavs, 1, threads, cb, cbSize);
    return outBuf;
}

// ── Fused: CreateBuffer + RunShader with 2 UAV outputs ──
extern "C" __declspec(dllexport) void* RunShaderAlloc2(const char* name, void** srvs, uint32_t srvCount,
    uint32_t outCount, void* extraUav, uint32_t* threads, void* cb, uint32_t cbSize) {
    void* outBuf = CreateBuffer(nullptr, outCount);
    if (!outBuf) return nullptr;
    void* uavs[2] = { outBuf, extraUav };
    RunShader(name, srvs, srvCount, uavs, 2, threads, cb, cbSize);
    return outBuf;
}

void Unik(int* number_to_dealloc) {
    *number_to_dealloc = *number_to_dealloc * 2 * sqrt(int(*number_to_dealloc));
}


// ── Batched SGD: all param updates in one call ──
struct SGDParams { uint32_t count; float lr; float clip; uint32_t pad; };

extern "C" __declspec(dllexport) void SGDBatch(void** params, void** grads, uint32_t* sizes,
    uint32_t numParams, float lr, float clip) {
    if (!g_context) return;
    auto it = g_shaders.find("sgd");
    if (it == g_shaders.end()) return;
    ID3D11ComputeShader* s = it->second;

    SGDParams sp;
    sp.lr = lr;
    sp.clip = clip;
    sp.pad = 0;

    uint32_t alignedSize = 16;
    if (!g_cbCache || g_cbCacheSize < alignedSize) {
        if (g_cbCache) g_cbCache->Release();
        D3D11_BUFFER_DESC cbd = { alignedSize, D3D11_USAGE_DYNAMIC, D3D11_BIND_CONSTANT_BUFFER, D3D11_CPU_ACCESS_WRITE, 0, 0 };
        g_device->CreateBuffer(&cbd, nullptr, &g_cbCache);
        g_cbCacheSize = alignedSize;
    }

    // Set shader once for all parameters (same kernel)
    g_context->CSSetShader(s, nullptr, 0);
    g_context->CSSetConstantBuffers(0, 1, &g_cbCache);

    for (uint32_t i = 0; i < numParams; i++) {
        if (!params[i] || !grads[i]) continue;
        GPUBuffer* grad = (GPUBuffer*)grads[i];
        GPUBuffer* param = (GPUBuffer*)params[i];

        ID3D11ShaderResourceView* srvPtrs[1] = { grad->srv };
        g_context->CSSetShaderResources(0, 1, srvPtrs);

        ID3D11UnorderedAccessView* uavPtrs[1] = { param->uav };
        g_context->CSSetUnorderedAccessViews(0, 1, uavPtrs, nullptr);

        sp.count = sizes[i];
        D3D11_MAPPED_SUBRESOURCE m;
        if (SUCCEEDED(g_context->Map(g_cbCache, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
            memcpy(m.pData, &sp, sizeof(sp));
            g_context->Unmap(g_cbCache, 0);
        }

        g_context->Dispatch((sizes[i] + 255) / 256, 1, 1);
    }
    // Single unbind after all SGD updates
    g_context->CSSetShaderResources(0, 1, g_nullSRVs);
    g_context->CSSetUnorderedAccessViews(0, 1, g_nullUAVs, nullptr);
    g_prevSrvCount = 1;
    g_prevUavCount = 1;
}

// ── Batched SGD with GPU-side gradient clipping ──
// Computes global grad norm entirely on GPU, clips, then does SGD.
// Returns the total grad norm (1 float readback instead of N full-param readbacks).
struct GradReduceParams { uint32_t count; uint32_t globalOffset; };
struct NormFinalParams { uint32_t totalPartials; uint32_t maxNormBits; };
struct SGDClipParams { uint32_t count; uint32_t lrBits; };

extern "C" __declspec(dllexport) float SGDBatchClipped(void** params, void** grads, uint32_t* sizes,
    uint32_t numParams, float lr, float maxNorm) {
    if (!g_context || numParams == 0) return 0.0f;

    auto itReduce = g_shaders.find("grad_sq_reduce");
    auto itFinal  = g_shaders.find("grad_norm_final");
    auto itSGD    = g_shaders.find("sgd_clipped");
    if (itReduce == g_shaders.end() || itFinal == g_shaders.end() || itSGD == g_shaders.end())
        return 0.0f;

    // Calculate total partial groups needed
    uint32_t totalPartials = 0;
    for (uint32_t i = 0; i < numParams; i++)
        totalPartials += (sizes[i] + 255) / 256;

    // Allocate temp buffers
    void* partials  = CreateBuffer(nullptr, totalPartials);
    void* normScale = CreateBuffer(nullptr, 2);  // [norm, scale]
    if (!partials || !normScale) return 0.0f;

    // ── Phase 1: Compute per-group grad² partial sums ──
    ID3D11ComputeShader* reduceShader = itReduce->second;
    g_context->CSSetShader(reduceShader, nullptr, 0);

    uint32_t alignedCB = 16;
    if (!g_cbCache || g_cbCacheSize < alignedCB) {
        if (g_cbCache) g_cbCache->Release();
        D3D11_BUFFER_DESC cbd = { alignedCB, D3D11_USAGE_DYNAMIC, D3D11_BIND_CONSTANT_BUFFER, D3D11_CPU_ACCESS_WRITE, 0, 0 };
        g_device->CreateBuffer(&cbd, nullptr, &g_cbCache);
        g_cbCacheSize = alignedCB;
    }

    ID3D11UnorderedAccessView* partialUAV[1] = { ((GPUBuffer*)partials)->uav };
    g_context->CSSetUnorderedAccessViews(0, 1, partialUAV, nullptr);

    uint32_t offset = 0;
    for (uint32_t i = 0; i < numParams; i++) {
        if (!grads[i]) continue;
        GPUBuffer* grad = (GPUBuffer*)grads[i];
        ID3D11ShaderResourceView* srvPtrs[1] = { grad->srv };
        g_context->CSSetShaderResources(0, 1, srvPtrs);

        GradReduceParams rp = { sizes[i], offset };
        D3D11_MAPPED_SUBRESOURCE m;
        if (SUCCEEDED(g_context->Map(g_cbCache, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
            memcpy(m.pData, &rp, sizeof(rp));
            g_context->Unmap(g_cbCache, 0);
            g_context->CSSetConstantBuffers(0, 1, &g_cbCache);
        }
        uint32_t groups = (sizes[i] + 255) / 256;
        g_context->Dispatch(groups, 1, 1);
        offset += groups;
    }

    // Unbind phase 1
    g_context->CSSetShaderResources(0, 1, g_nullSRVs);
    g_context->CSSetUnorderedAccessViews(0, 1, g_nullUAVs, nullptr);

    // ── Phase 2: Reduce all partials → total norm + clip scale ──
    g_context->CSSetShader(itFinal->second, nullptr, 0);
    {
        ID3D11ShaderResourceView* srvPtrs[1] = { ((GPUBuffer*)partials)->srv };
        g_context->CSSetShaderResources(0, 1, srvPtrs);
        ID3D11UnorderedAccessView* uavPtrs[1] = { ((GPUBuffer*)normScale)->uav };
        g_context->CSSetUnorderedAccessViews(0, 1, uavPtrs, nullptr);

        uint32_t maxNormBits;
        memcpy(&maxNormBits, &maxNorm, 4);
        NormFinalParams fp = { totalPartials, maxNormBits };
        D3D11_MAPPED_SUBRESOURCE m;
        if (SUCCEEDED(g_context->Map(g_cbCache, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
            memcpy(m.pData, &fp, sizeof(fp));
            g_context->Unmap(g_cbCache, 0);
            g_context->CSSetConstantBuffers(0, 1, &g_cbCache);
        }
        g_context->Dispatch(1, 1, 1);
    }
    g_context->CSSetShaderResources(0, 1, g_nullSRVs);
    g_context->CSSetUnorderedAccessViews(0, 1, g_nullUAVs, nullptr);

    // ── Phase 3: SGD with clip scale from GPU buffer ──
    g_context->CSSetShader(itSGD->second, nullptr, 0);

    uint32_t lrBits;
    memcpy(&lrBits, &lr, 4);

    for (uint32_t i = 0; i < numParams; i++) {
        if (!params[i] || !grads[i]) continue;
        GPUBuffer* grad = (GPUBuffer*)grads[i];
        GPUBuffer* param = (GPUBuffer*)params[i];

        ID3D11ShaderResourceView* srvPtrs[2] = { grad->srv, ((GPUBuffer*)normScale)->srv };
        g_context->CSSetShaderResources(0, 2, srvPtrs);
        ID3D11UnorderedAccessView* uavPtrs[1] = { param->uav };
        g_context->CSSetUnorderedAccessViews(0, 1, uavPtrs, nullptr);

        SGDClipParams sp = { sizes[i], lrBits };
        D3D11_MAPPED_SUBRESOURCE m;
        if (SUCCEEDED(g_context->Map(g_cbCache, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
            memcpy(m.pData, &sp, sizeof(sp));
            g_context->Unmap(g_cbCache, 0);
            g_context->CSSetConstantBuffers(0, 1, &g_cbCache);
        }
        g_context->Dispatch((sizes[i] + 255) / 256, 1, 1);
    }

    // Final unbind
    g_context->CSSetShaderResources(0, 2, g_nullSRVs);
    g_context->CSSetUnorderedAccessViews(0, 1, g_nullUAVs, nullptr);
    g_prevSrvCount = 2;
    g_prevUavCount = 1;

    // Read back just the norm (1 tiny GPU→CPU read instead of 30 full params)
    float normData[2] = {0.0f, 0.0f};
    ReadBuffer(normScale, normData);
    float totalNorm = normData[0];

    ReleaseBuffer(partials);
    ReleaseBuffer(normScale);
    return totalNorm;
}

// ── Pool Warm-up: pre-create buffers and return them to pool ──
extern "C" __declspec(dllexport) void WarmPool(uint32_t* sizes, uint32_t numSizes, uint32_t copies) {
    if (!g_device) return;
    for (uint32_t i = 0; i < numSizes; i++) {
        uint32_t count = sizes[i];
        size_t maxPool = MaxPoolForSize(count);
        auto& pool = g_bufferPool[count];
        uint32_t target = copies < (uint32_t)maxPool ? copies : (uint32_t)maxPool;
        uint32_t needed = (target > (uint32_t)pool.size()) ? (target - (uint32_t)pool.size()) : 0;
        for (uint32_t c = 0; c < needed; c++) {
            GPUBuffer* b = (GPUBuffer*)CreateBuffer(nullptr, count);
            if (b) {
                ReleaseBufferInternal(b);
                // Don't count warm-up allocs as misses
                if (g_poolMisses > 0) g_poolMisses--;
            }
        }
    }
}

// ── Pool Stats ──
extern "C" __declspec(dllexport) void GetPoolStats(uint64_t* hits, uint64_t* misses) {
    if (hits) *hits = g_poolHits;
    if (misses) *misses = g_poolMisses;
}

extern "C" __declspec(dllexport) void ResetPoolStats() {
    g_poolHits = 0;
    g_poolMisses = 0;
}

// Return total bytes held in pool
extern "C" __declspec(dllexport) uint64_t GetPoolMemory() {
    return g_poolMemBytes;
}

// Drain entire pool — free all pooled buffers
extern "C" __declspec(dllexport) void DrainPool() {
    for (auto& kv : g_bufferPool) {
        for (GPUBuffer* b : kv.second) {
            if (b->srv) b->srv->Release();
            if (b->uav) b->uav->Release();
            if (b->buffer) b->buffer->Release();
            delete b;
        }
        kv.second.clear();
    }
    g_bufferPool.clear();
    g_poolMemBytes = 0;
    // Also drain staging pool
    for (auto& kv : g_stagingPool) {
        for (ID3D11Buffer* s : kv.second) {
            if (s) s->Release();
        }
        kv.second.clear();
    }
    g_stagingPool.clear();
}

// ── Batch release ──
extern "C" __declspec(dllexport) void ReleaseBufferBatch(void** handles, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        if (handles[i]) {
            GPUBuffer* b = (GPUBuffer*)handles[i];
            if (--b->refCount == 0) {
                ReleaseBufferInternal(b);
            }
        }
    }
}

// ── Fused Operations: reduce Python→C++ DLL crossings ──

// Internal matmul dispatch — replicates Python _run_mm logic
static void MatMulDispatch(void* a_buf, void* b_buf, void* out_buf,
                           uint32_t M, uint32_t K, uint32_t N, uint32_t flags) {
    const char* shaderName = g_gpuInfo.isIntegrated ? "matmul_coarsened" : "matmul_dgpu";
    uint32_t tileC = 64;  // Both iGPU and dGPU shaders now use TS=64

    uint32_t cbData[4] = {M, K, N, flags};
    void* srvs[2] = {a_buf, b_buf};
    void* uavs[1] = {out_buf};

    // Use parallel-reduction kernel ONLY when output is extremely small.
    // matmul_reduce dispatches (N, M, 1) = M*N groups, each with 256 threads.
    // This is ONLY beneficial when M*N is tiny (e.g. Conv1 dFilter: 6*25=150 groups).
    // For larger M*N (e.g. UNet up1: 16*432=6912 groups), the overhead of
    // 6912 groups × 256 threads × 1KB shared mem + 8 barriers = catastrophic!
    // Threshold: output_groups < 8 catches M*N < ~1024 (safe), K > 512 ensures benefit.
    uint32_t output_groups = ((N + 15) / 16) * ((M + 15) / 16);
    if (output_groups < 8 && K > 512) {
        // Use matmul_reduce: one group per output element, 256 threads reduce K
        // Best for conv backward dFilter where M×N is tiny but K is huge
        uint32_t threads[3] = {N, M, 1};
        auto it = g_shaders.find("matmul_reduce");
        if (it != g_shaders.end()) {
            DispatchInternal(it->second, srvs, 2, uavs, 1, threads, cbData, 16);
            return;
        }
    }

    if (M >= 64 && N >= 64) {
        uint32_t threads[3] = {(N + tileC - 1) / tileC, (M + tileC - 1) / tileC, 1};
        auto it = g_shaders.find(shaderName);
        if (it != g_shaders.end())
            DispatchInternal(it->second, srvs, 2, uavs, 1, threads, cbData, 16);
    } else {
        uint32_t threads[3] = {(N + 15) / 16, (M + 15) / 16, 1};
        auto it = g_shaders.find("matmul_universal");
        if (it != g_shaders.end())
            DispatchInternal(it->second, srvs, 2, uavs, 1, threads, cbData, 16);
    }
}

// Fused matmul: CreateBuffer + matmul dispatch in one call
extern "C" __declspec(dllexport) void* MatMulAlloc(
    void* a_buf, void* b_buf, uint32_t M, uint32_t K, uint32_t N, uint32_t flags) {
    void* out = CreateBuffer(nullptr, M * N);
    if (!out) return nullptr;
    MatMulDispatch(a_buf, b_buf, out, M, K, N, flags);
    return out;
}

// Fused ConvForward: im2col + matmul + conv_reshape in one C++ call
// Returns output buffer handle. Stores im2col handle in *pIm2col for backward pass.
extern "C" __declspec(dllexport) void* ConvForwardFused(
    void* x_buf, void* filters_buf, void* bias_buf,
    uint32_t batch, uint32_t inC, uint32_t inH, uint32_t inW,
    uint32_t outC, uint32_t kH, uint32_t kW,
    uint32_t stride, uint32_t padding, uint32_t actType,
    void** pIm2col) {

    uint32_t outH = (inH + 2 * padding - kH) / stride + 1;
    uint32_t outW = (inW + 2 * padding - kW) / stride + 1;
    uint32_t totalRow = inC * kH * kW;
    uint32_t totalCol = batch * outH * outW;

    // 1. im2col
    void* im2col = CreateBuffer(nullptr, totalRow * totalCol);
    if (!im2col) return nullptr;
    {
        uint32_t cb[9] = {batch, inC, inH, inW, kH, stride, padding, outH, outW};
        void* srvs[1] = {x_buf};
        void* uavs[1] = {im2col};
        uint32_t threads[3] = {(totalCol + 15) / 16, (totalRow + 15) / 16, 1};
        auto it = g_shaders.find("im2col");
        if (it != g_shaders.end())
            DispatchInternal(it->second, srvs, 1, uavs, 1, threads, cb, 36);
    }

    // 2. matmul: filters × im2col
    void* mm = CreateBuffer(nullptr, outC * totalCol);
    if (!mm) { ReleaseBuffer(im2col); return nullptr; }
    MatMulDispatch(filters_buf, im2col, mm, outC, totalRow, totalCol, 0);

    // 3. conv_reshape + bias + optional relu
    void* out = CreateBuffer(nullptr, batch * outC * outH * outW);
    if (!out) { ReleaseBuffer(im2col); ReleaseBuffer(mm); return nullptr; }
    {
        uint32_t cb[5] = {batch, outC, outH, outW, actType};
        void* srvs[2] = {mm, bias_buf};
        void* uavs[1] = {out};
        uint32_t threads[3] = {(totalCol + 15) / 16, (outC + 15) / 16, 1};
        auto it = g_shaders.find("conv_reshape");
        if (it != g_shaders.end())
            DispatchInternal(it->second, srvs, 2, uavs, 1, threads, cb, 20);
    }

    ReleaseBuffer(mm);
    if (pIm2col) *pIm2col = im2col;
    else ReleaseBuffer(im2col);
    return out;
}

// Fused ConvBackward: all backward dispatches in one C++ call
// Returns gradient handles via output pointers. Releases grad_reshaped and im2col internally.
extern "C" __declspec(dllexport) void ConvBackwardFused(
    void* grad_output_buf, void* fwd_output_buf,
    void* x_buf, void* filters_buf, void* im2col_buf,
    uint32_t batch, uint32_t inC, uint32_t inH, uint32_t inW,
    uint32_t outC, uint32_t kH, uint32_t kW,
    uint32_t stride, uint32_t padding, uint32_t actType,
    uint32_t x_requires_grad, uint32_t filters_requires_grad, uint32_t bias_requires_grad,
    void** pDFilters, void** pDBias, void** pDInput) {

    uint32_t outH = (inH + 2 * padding - kH) / stride + 1;
    uint32_t outW = (inW + 2 * padding - kW) / stride + 1;
    uint32_t totalRow = inC * kH * kW;
    uint32_t totalCol = batch * outH * outW;

    // 1. grad_reshaped (fused with relu grad if applicable)
    void* grad_reshaped = CreateBuffer(nullptr, outC * totalCol);
    {
        uint32_t cb[9] = {batch, 0, 0, 0, outC, 0, 0, outH, outW};
        if (actType == 1 && fwd_output_buf) {
            void* srvs[2] = {grad_output_buf, fwd_output_buf};
            void* uavs[1] = {grad_reshaped};
            uint32_t threads[3] = {(totalCol + 15) / 16, (outC + 15) / 16, 1};
            auto it = g_shaders.find("conv_grad_reshape_relu");
            if (it != g_shaders.end())
                DispatchInternal(it->second, srvs, 2, uavs, 1, threads, cb, 36);
        } else {
            void* srvs[1] = {grad_output_buf};
            void* uavs[1] = {grad_reshaped};
            uint32_t threads[3] = {(totalCol + 15) / 16, (outC + 15) / 16, 1};
            auto it = g_shaders.find("conv_grad_reshape");
            if (it != g_shaders.end())
                DispatchInternal(it->second, srvs, 1, uavs, 1, threads, cb, 36);
        }
    }

    // 2. dFilters = grad_reshaped × im2col^T
    if (filters_requires_grad && pDFilters) {
        void* dF = CreateBuffer(nullptr, outC * totalRow);
        MatMulDispatch(grad_reshaped, im2col_buf, dF, outC, totalCol, totalRow, 2);
        *pDFilters = dF;
    } else if (pDFilters) *pDFilters = nullptr;

    // 3. dBias
    if (bias_requires_grad && pDBias) {
        void* dB = CreateBuffer(nullptr, outC);
        uint32_t cb[4] = {totalCol, outC, 1, totalCol};
        void* srvs[1] = {grad_reshaped};
        void* uavs[1] = {dB};
        uint32_t threads[3] = {outC, 1, 1};
        auto it = g_shaders.find("bias_grad");
        if (it != g_shaders.end())
            DispatchInternal(it->second, srvs, 1, uavs, 1, threads, cb, 16);
        *pDBias = dB;
    } else if (pDBias) *pDBias = nullptr;

    // 4. dInput
    if (x_requires_grad && pDInput) {
        void* dIcol = CreateBuffer(nullptr, totalRow * totalCol);
        MatMulDispatch(filters_buf, grad_reshaped, dIcol, totalRow, outC, totalCol, 1);

        uint32_t xSize = batch * inC * inH * inW;
        void* dIn = CreateBuffer(nullptr, xSize);
        uint32_t cb[9] = {batch, inC, inH, inW, kH, stride, padding, outH, outW};
        void* srvs[1] = {dIcol};
        void* uavs[1] = {dIn};
        uint32_t threads[3] = {(xSize + 255) / 256, 1, 1};
        auto it = g_shaders.find("col2im");
        if (it != g_shaders.end())
            DispatchInternal(it->second, srvs, 1, uavs, 1, threads, cb, 36);
        ReleaseBuffer(dIcol);
        *pDInput = dIn;
    } else if (pDInput) *pDInput = nullptr;

    // Cleanup
    ReleaseBuffer(grad_reshaped);
    ReleaseBuffer(im2col_buf);
}

// Fused MaxPool forward: CreateBuffer×2 + dispatch in one call
extern "C" __declspec(dllexport) void* MaxPoolForwardFused(
    void* x_buf, uint32_t batch, uint32_t inC, uint32_t inH, uint32_t inW,
    uint32_t pool_size, uint32_t stride, void** pIndices) {

    uint32_t outH = (inH - pool_size) / stride + 1;
    uint32_t outW = (inW - pool_size) / stride + 1;
    uint32_t outSize = batch * inC * outH * outW;

    void* out = CreateBuffer(nullptr, outSize);
    void* indices = CreateBuffer(nullptr, outSize);
    if (!out || !indices) return nullptr;

    uint32_t cb[8] = {batch, inC, inH, inW, pool_size, stride, outH, outW};
    void* srvs[1] = {x_buf};
    void* uavs[2] = {out, indices};
    uint32_t threads[3] = {(outW + 15) / 16, (outH + 15) / 16, batch * inC};
    auto it = g_shaders.find("maxpool_forward");
    if (it != g_shaders.end())
        DispatchInternal(it->second, srvs, 1, uavs, 2, threads, cb, 32);

    if (pIndices) *pIndices = indices;
    return out;
}

// Fused MaxPool backward: CreateBuffer + Clear + dispatch + release indices
extern "C" __declspec(dllexport) void* MaxPoolBackwardFused(
    void* grad_output_buf, void* indices_buf,
    uint32_t batch, uint32_t inC, uint32_t outH, uint32_t outW,
    uint32_t x_size) {

    void* dInput = CreateBuffer(nullptr, x_size);
    if (!dInput) return nullptr;

    // Clear the buffer (col2im-style accumulation needs zeroed output)
    GPUBuffer* b = (GPUBuffer*)dInput;
    uint32_t clear[4] = {0, 0, 0, 0};
    if (b->uav) g_context->ClearUnorderedAccessViewUint(b->uav, clear);

    uint32_t cb[4] = {batch, inC, outH, outW};
    void* srvs[2] = {grad_output_buf, indices_buf};
    void* uavs[1] = {dInput};
    uint32_t threads[3] = {(outW + 15) / 16, (outH + 15) / 16, batch * inC};
    auto it = g_shaders.find("maxpool_backward");
    if (it != g_shaders.end())
        DispatchInternal(it->second, srvs, 2, uavs, 1, threads, cb, 16);

    // Release indices buffer (no longer needed after backward)
    ReleaseBuffer(indices_buf);
    return dInput;
}

// ── Graph Replay: resolve shader + execute pre-built dispatch sequences ──

extern "C" __declspec(dllexport) void* ResolveShader(const char* name) {
    auto it = g_shaders.find(name);
    return (it != g_shaders.end()) ? (void*)it->second : nullptr;
}

// Packed dispatch command — 256 bytes, cache-line friendly
struct SeqCmd {
    void* shader;            // pre-resolved ID3D11ComputeShader*
    void* srvs[8];           // buffer handles (GPUBuffer*)
    uint32_t srvCount;
    void* uavs[4];           // buffer handles (GPUBuffer*)
    uint32_t uavCount;
    uint32_t threads[3];
    uint8_t cbData[64];      // constant buffer data (inline)
    uint32_t cbSize;
    uint32_t pad[1];         // align to nice boundary
};

extern "C" __declspec(dllexport) void RunSequence(SeqCmd* cmds, uint32_t count) {
    if (!g_context || !cmds) return;
    for (uint32_t i = 0; i < count; i++) {
        SeqCmd& cmd = cmds[i];
        if (!cmd.shader) continue;
        DispatchInternal(
            (ID3D11ComputeShader*)cmd.shader,
            cmd.srvs, cmd.srvCount,
            cmd.uavs, cmd.uavCount,
            cmd.threads,
            cmd.cbSize > 0 ? cmd.cbData : nullptr,
            cmd.cbSize
        );
    }
}

// Variant: RunSequence + SGDBatch all in one call (forward+backward+optimizer)
extern "C" __declspec(dllexport) void RunSequenceWithSGD(
    SeqCmd* cmds, uint32_t cmdCount,
    void** params, void** grads, uint32_t* sizes, uint32_t numParams,
    float lr, float clip) {
    // Execute all recorded dispatches (forward + backward)
    RunSequence(cmds, cmdCount);
    // Then SGD update
    SGDBatch(params, grads, sizes, numParams, lr, clip);
}
