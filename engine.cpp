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

static size_t g_maxPoolPerSize = 8;

extern "C" __declspec(dllexport) void* CreateBuffer(float* data, uint32_t count) {
    if (!g_device) return nullptr;
    GPUBuffer* b = nullptr;

    auto it = g_bufferPool.find(count);
    if (it != g_bufferPool.end() && !it->second.empty()) {
        b = it->second.back();
        it->second.pop_back();
        b->refCount = 1;
        if (data) {
            g_context->UpdateSubresource(b->buffer, 0, nullptr, data, 0, 0);
        }
        return b;
    }

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
    if (pool.size() < g_maxPoolPerSize) {
        pool.push_back(b);
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
