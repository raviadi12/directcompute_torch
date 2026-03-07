#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <iostream>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

struct Constants {
    uint32_t widthA;
    uint32_t heightA;
    uint32_t widthB;
    uint32_t heightB;
};

ID3D11Device* g_device = nullptr;
ID3D11DeviceContext* g_context = nullptr;
ID3D11ComputeShader* g_shader = nullptr;

extern "C" __declspec(dllexport) bool InitDirectCompute(const wchar_t* shaderPath) {
    if (g_device) return true;
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &g_device, &featureLevel, &g_context);
    if (FAILED(hr)) return false;

    ID3DBlob* shaderBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;
    hr = D3DCompileFromFile(shaderPath, nullptr, nullptr, "CSMain", "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &shaderBlob, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) {
            std::cerr << "DirectCompute Shader Compile Error: " << (char*)errorBlob->GetBufferPointer() << std::endl;
            errorBlob->Release();
        }
        return false;
    }
    g_device->CreateComputeShader(shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(), nullptr, &g_shader);
    shaderBlob->Release();
    return g_shader != nullptr;
}

extern "C" __declspec(dllexport) void CleanupDirectCompute() {
    if (g_shader) { g_shader->Release(); g_shader = nullptr; }
    if (g_context) { g_context->Release(); g_context = nullptr; }
    if (g_device) { g_device->Release(); g_device = nullptr; }
}

extern "C" __declspec(dllexport) bool MatMul(const float* A, const float* B, float* C, int M, int K, int N) {
    if (!g_device || !g_shader) return false;

    auto CreateBuffer = [&](const float* data, size_t size, UINT bindFlags, D3D11_USAGE usage, UINT cpuAccess) -> ID3D11Buffer* {
        D3D11_BUFFER_DESC desc = {};
        desc.Usage = usage;
        desc.ByteWidth = sizeof(float) * size;
        desc.BindFlags = bindFlags;
        desc.CPUAccessFlags = cpuAccess;
        if (bindFlags & D3D11_BIND_SHADER_RESOURCE || bindFlags & D3D11_BIND_UNORDERED_ACCESS)
            desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(float);

        D3D11_SUBRESOURCE_DATA initData = {};
        if (data) {
            initData.pSysMem = data;
        }

        ID3D11Buffer* buffer = nullptr;
        g_device->CreateBuffer(&desc, data ? &initData : nullptr, &buffer);
        return buffer;
    };

    ID3D11Buffer* bufA = CreateBuffer(A, M * K, D3D11_BIND_SHADER_RESOURCE, D3D11_USAGE_DEFAULT, 0);
    ID3D11Buffer* bufB = CreateBuffer(B, K * N, D3D11_BIND_SHADER_RESOURCE, D3D11_USAGE_DEFAULT, 0);
    ID3D11Buffer* bufC = CreateBuffer(nullptr, M * N, D3D11_BIND_UNORDERED_ACCESS, D3D11_USAGE_DEFAULT, 0);
    ID3D11Buffer* stagingC = CreateBuffer(nullptr, M * N, 0, D3D11_USAGE_STAGING, D3D11_CPU_ACCESS_READ);

    auto CreateSRV = [&](ID3D11Buffer* buffer, size_t size) -> ID3D11ShaderResourceView* {
        D3D11_SHADER_RESOURCE_VIEW_DESC desc = {};
        desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
        desc.BufferEx.NumElements = size;
        desc.Format = DXGI_FORMAT_UNKNOWN;
        ID3D11ShaderResourceView* srv = nullptr;
        g_device->CreateShaderResourceView(buffer, &desc, &srv);
        return srv;
    };

    ID3D11ShaderResourceView* srvA = CreateSRV(bufA, M * K);
    ID3D11ShaderResourceView* srvB = CreateSRV(bufB, K * N);

    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    uavDesc.Buffer.NumElements = M * N;
    uavDesc.Format = DXGI_FORMAT_UNKNOWN;
    ID3D11UnorderedAccessView* uavC = nullptr;
    g_device->CreateUnorderedAccessView(bufC, &uavDesc, &uavC);

    Constants consts = { (uint32_t)K, (uint32_t)M, (uint32_t)N, (uint32_t)K };
    D3D11_BUFFER_DESC cbDesc = {};
    cbDesc.Usage = D3D11_USAGE_DEFAULT;
    cbDesc.ByteWidth = sizeof(Constants);
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA cbData = {};
    cbData.pSysMem = &consts;
    ID3D11Buffer* cb = nullptr;
    g_device->CreateBuffer(&cbDesc, &cbData, &cb);

    g_context->CSSetShader(g_shader, nullptr, 0);
    g_context->CSSetConstantBuffers(0, 1, &cb);
    ID3D11ShaderResourceView* srvs[] = { srvA, srvB };
    g_context->CSSetShaderResources(0, 2, srvs);
    g_context->CSSetUnorderedAccessViews(0, 1, &uavC, nullptr);

    // 2D Coarsened processes 64 columns and 64 rows per Thread Group
    g_context->Dispatch((N + 63) / 64, (M + 63) / 64, 1);

    g_context->CopyResource(stagingC, bufC);
    
    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hr = g_context->Map(stagingC, 0, D3D11_MAP_READ, 0, &mapped);
    if (SUCCEEDED(hr)) {
        memcpy(C, mapped.pData, sizeof(float) * M * N);
        g_context->Unmap(stagingC, 0);
    }

    // Unbind
    ID3D11UnorderedAccessView* nullUAV[] = {nullptr};
    g_context->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);
    ID3D11ShaderResourceView* nullSRV[] = {nullptr, nullptr};
    g_context->CSSetShaderResources(0, 2, nullSRV);

    if (cb) cb->Release(); 
    if (uavC) uavC->Release(); 
    if (srvB) srvB->Release(); 
    if (srvA) srvA->Release();
    if (stagingC) stagingC->Release(); 
    if (bufC) bufC->Release(); 
    if (bufB) bufB->Release(); 
    if (bufA) bufA->Release();

    return SUCCEEDED(hr);
}
