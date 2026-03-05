#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <omp.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

struct Constants {
    uint32_t widthA;
    uint32_t heightA;
    uint32_t widthB;
    uint32_t heightB;
};

void MatMulCPU_Single(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void MatMulCPU_Multi(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int K, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

bool Verify(const std::vector<float>& ref, const std::vector<float>& test, float threshold = 1e-2f) {
    if (ref.size() != test.size()) return false;
    for (size_t i = 0; i < ref.size(); ++i) {
        // use relative error for larger values or simply higher threshold due to float precision accumulation
        if (std::abs(ref[i] - test[i]) > threshold && std::abs((ref[i] - test[i])/ref[i]) > 1e-4f ) {
            std::cerr << "Mismatch at " << i << ": ref " << ref[i] << " vs test " << test[i] << std::endl;
            return false;
        }
    }
    return true;
}

ID3D11ComputeShader* CompileShader(ID3D11Device* device, const wchar_t* filename) {
    ID3DBlob* shaderBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;
    HRESULT hr = D3DCompileFromFile(filename, nullptr, nullptr, "CSMain", "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &shaderBlob, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) {
            std::cerr << "Shader compile error (" << filename << "): " << (char*)errorBlob->GetBufferPointer() << std::endl;
            errorBlob->Release();
        }
        return nullptr;
    }
    ID3D11ComputeShader* computeShader = nullptr;
    device->CreateComputeShader(shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(), nullptr, &computeShader);
    shaderBlob->Release();
    return computeShader;
}

void RunBenchmark(int SIZE) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmarking Matrix Size: " << SIZE << " x " << SIZE << std::endl;
    std::cout << "========================================" << std::endl;

    int M = SIZE, K = SIZE, N = SIZE;
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_ref(M * N, 0.0f);
    std::vector<float> C_test(M * N, 0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (auto& x : A) x = dis(gen);
    for (auto& x : B) x = dis(gen);

    // CPU Single Core
    std::cout << "Running CPU Single-core..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    MatMulCPU_Single(A, B, C_ref, M, K, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_single_time = end - start;
    std::cout << "CPU Single-core Time: " << cpu_single_time.count() << " ms" << std::endl;

    // CPU Multi Core
    std::cout << "Running CPU Multi-core..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    MatMulCPU_Multi(A, B, C_test, M, K, N);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_multi_time = end - start;
    std::cout << "CPU Multi-core Time:  " << cpu_multi_time.count() << " ms" << std::endl;
    std::cout << "CPU Multi-core Speedup: " << cpu_single_time.count() / cpu_multi_time.count() << "x" << std::endl;
    if (Verify(C_ref, C_test)) {
        std::cout << "CPU Multi-core Result: VERIFIED" << std::endl;
    } else {
        std::cout << "CPU Multi-core Result: FAILED" << std::endl;
    }

    // DirectCompute Setup
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, &featureLevel, &context);
    if (FAILED(hr)) {
        std::cerr << "Failed to create D3D11 device." << std::endl;
        return;
    }

    ID3D11ComputeShader* basicShader = CompileShader(device, L"matmul.hlsl");
    ID3D11ComputeShader* tiledShader = CompileShader(device, L"matmul_tiled.hlsl");
    ID3D11ComputeShader* tiledUnrollShader = CompileShader(device, L"matmul_tiled_unroll.hlsl");
    ID3D11ComputeShader* coarsenedShader = CompileShader(device, L"matmul_coarsened.hlsl");
    ID3D11ComputeShader* coarsened2DShader = CompileShader(device, L"matmul_coarsened_2d.hlsl");

    if (!basicShader || !tiledShader || !tiledUnrollShader || !coarsenedShader || !coarsened2DShader) {
        std::cerr << "Failed to compile shaders." << std::endl;
        return;
    }

    // Create Buffers
    auto CreateStructuredBuffer = [&](const std::vector<float>& data) -> ID3D11Buffer* {
        D3D11_BUFFER_DESC desc = {};
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.ByteWidth = sizeof(float) * data.size();
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(float);

        D3D11_SUBRESOURCE_DATA initData = {};
        initData.pSysMem = data.data();

        ID3D11Buffer* buffer = nullptr;
        device->CreateBuffer(&desc, &initData, &buffer);
        return buffer;
    };

    auto CreateSRV = [&](ID3D11Buffer* buffer) -> ID3D11ShaderResourceView* {
        D3D11_SHADER_RESOURCE_VIEW_DESC desc = {};
        desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
        desc.BufferEx.FirstElement = 0;
        D3D11_BUFFER_DESC bufDesc;
        buffer->GetDesc(&bufDesc);
        desc.BufferEx.NumElements = bufDesc.ByteWidth / bufDesc.StructureByteStride;
        desc.Format = DXGI_FORMAT_UNKNOWN;

        ID3D11ShaderResourceView* srv = nullptr;
        device->CreateShaderResourceView(buffer, &desc, &srv);
        return srv;
    };

    ID3D11Buffer* bufferA = CreateStructuredBuffer(A);
    ID3D11Buffer* bufferB = CreateStructuredBuffer(B);
    ID3D11ShaderResourceView* srvA = CreateSRV(bufferA);
    ID3D11ShaderResourceView* srvB = CreateSRV(bufferB);

    // Create Output Buffer (UAV)
    D3D11_BUFFER_DESC outDesc = {};
    outDesc.Usage = D3D11_USAGE_DEFAULT;
    outDesc.ByteWidth = sizeof(float) * M * N;
    outDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    outDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    outDesc.StructureByteStride = sizeof(float);
    ID3D11Buffer* bufferOut = nullptr;
    device->CreateBuffer(&outDesc, nullptr, &bufferOut);

    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    uavDesc.Buffer.FirstElement = 0;
    uavDesc.Buffer.NumElements = M * N;
    uavDesc.Format = DXGI_FORMAT_UNKNOWN;
    ID3D11UnorderedAccessView* uavOut = nullptr;
    device->CreateUnorderedAccessView(bufferOut, &uavDesc, &uavOut);

    // Constant Buffer
    Constants consts = { (uint32_t)K, (uint32_t)M, (uint32_t)N, (uint32_t)K };
    D3D11_BUFFER_DESC cbDesc = {};
    cbDesc.Usage = D3D11_USAGE_DEFAULT;
    cbDesc.ByteWidth = sizeof(Constants);
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA cbData = {};
    cbData.pSysMem = &consts;
    ID3D11Buffer* constBuffer = nullptr;
    device->CreateBuffer(&cbDesc, &cbData, &constBuffer);

    // Staging Buffer for reading results
    D3D11_BUFFER_DESC stagingDesc = {};
    stagingDesc.Usage = D3D11_USAGE_STAGING;
    stagingDesc.ByteWidth = sizeof(float) * M * N;
    stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Buffer* stagingBuffer = nullptr;
    device->CreateBuffer(&stagingDesc, nullptr, &stagingBuffer);

    // Setup Compute Pipeline
    context->CSSetConstantBuffers(0, 1, &constBuffer);
    ID3D11ShaderResourceView* srvs[] = { srvA, srvB };
    context->CSSetShaderResources(0, 2, srvs);
    context->CSSetUnorderedAccessViews(0, 1, &uavOut, nullptr);

    // Print Adapter Name
    IDXGIDevice* dxgiDevice = nullptr;
    if (SUCCEEDED(device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice))) {
        IDXGIAdapter* dxgiAdapter = nullptr;
        if (SUCCEEDED(dxgiDevice->GetAdapter(&dxgiAdapter))) {
            DXGI_ADAPTER_DESC adapterDesc;
            dxgiAdapter->GetDesc(&adapterDesc);
            std::wcout << L"Using GPU Adapter: " << adapterDesc.Description << std::endl;
            dxgiAdapter->Release();
        }
        dxgiDevice->Release();
    }

    // Helper to run and time shader
    auto RunComputeShader = [&](ID3D11ComputeShader* shader, const char* name, int dispatchX = -1, int dispatchY = -1) {
        if (dispatchX == -1) dispatchX = (N + 15) / 16;
        if (dispatchY == -1) dispatchY = (M + 15) / 16;
        
        context->CSSetShader(shader, nullptr, 0);
        
        // Warmup
        context->Dispatch(dispatchX, dispatchY, 1);
        
        // Wait for warmup to finish by doing a dummy map
        context->CopyResource(stagingBuffer, bufferOut);
        D3D11_MAPPED_SUBRESOURCE warmupMapped;
        context->Map(stagingBuffer, 0, D3D11_MAP_READ, 0, &warmupMapped);
        context->Unmap(stagingBuffer, 0);

        std::cout << "Running GPU " << name << "..." << std::endl;
        
        auto start_gpu = std::chrono::high_resolution_clock::now();
        
        // Main execution
        context->Dispatch(dispatchX, dispatchY, 1);
        
        // Copy to staging buffer to wait for completion
        context->CopyResource(stagingBuffer, bufferOut);
        
        D3D11_MAPPED_SUBRESOURCE mappedResource;
        context->Map(stagingBuffer, 0, D3D11_MAP_READ, 0, &mappedResource);
        auto end_gpu = std::chrono::high_resolution_clock::now();
        
        float* dataPtr = (float*)mappedResource.pData;
        std::vector<float> gpu_res(dataPtr, dataPtr + M * N);
        context->Unmap(stagingBuffer, 0);

        std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;
        std::cout << "GPU " << name << " Time: " << gpu_time.count() << " ms" << std::endl;
        std::cout << "GPU " << name << " Speedup: " << cpu_single_time.count() / gpu_time.count() << "x" << std::endl;

        if (Verify(C_ref, gpu_res)) {
            std::cout << "GPU " << name << " Result: VERIFIED" << std::endl;
        } else {
            std::cout << "GPU " << name << " Result: FAILED" << std::endl;
        }
    };

    RunComputeShader(basicShader, "Basic");
    
    // Clear out uav to 0
    const float clearVals[4] = {0,0,0,0};
    context->ClearUnorderedAccessViewFloat(uavOut, clearVals);

    RunComputeShader(tiledShader, "Tiled");

    context->ClearUnorderedAccessViewFloat(uavOut, clearVals);

    RunComputeShader(tiledUnrollShader, "Tiled Unrolled");

    context->ClearUnorderedAccessViewFloat(uavOut, clearVals);
    
    // Coarsened processes 64 columns and 16 rows per Thread Group
    RunComputeShader(coarsenedShader, "Coarsened", (N + 63) / 64, (M + 15) / 16);

    context->ClearUnorderedAccessViewFloat(uavOut, clearVals);

    // Coarsened 2D processes 64 columns and 64 rows per Thread Group
    RunComputeShader(coarsened2DShader, "Coarsened 2D", (N + 63) / 64, (M + 63) / 64);

    // Cleanup
    srvA->Release(); srvB->Release();
    bufferA->Release(); bufferB->Release();
    bufferOut->Release(); uavOut->Release();
    constBuffer->Release(); stagingBuffer->Release();
    basicShader->Release(); tiledShader->Release(); 
    tiledUnrollShader->Release(); coarsenedShader->Release(); coarsened2DShader->Release();
    context->Release(); device->Release();
}

int main() {
    // Run for a small size to warm up and verify
    RunBenchmark(256);

    // Run for the requested 1024x1024 size
    RunBenchmark(1024);

    return 0;
}
