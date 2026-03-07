#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <iostream>
#include <chrono>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "ole32.lib")

// Config globals
int g_radius = 5;
float g_sigma = 3.0f;
bool g_running = true;

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_DESTROY) {
        g_running = false;
        PostQuitMessage(0);
        return 0;
    }
    if (msg == WM_KEYDOWN) {
        if (wParam == VK_UP) { 
            g_radius += 2; 
            g_sigma += 1.0f; 
            std::cout << "Increased Blur -> Radius: " << g_radius << " | Sigma: " << g_sigma << std::endl;
        }
        if (wParam == VK_DOWN) { 
            g_radius = max(1, g_radius - 2); 
            g_sigma = max(0.5f, g_sigma - 1.0f); 
            std::cout << "Decreased Blur -> Radius: " << g_radius << " | Sigma: " << g_sigma << std::endl;
        }
        if (wParam == VK_ESCAPE) { 
            g_running = false; 
            PostQuitMessage(0); 
        }
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

ID3D11ComputeShader* CompileComputeShader(ID3D11Device* device, const wchar_t* filename, const char* entryPoint) {
    ID3DBlob* shaderBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;
    HRESULT hr = D3DCompileFromFile(filename, nullptr, nullptr, entryPoint, "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &shaderBlob, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) {
            std::cerr << "Shader compile error: " << (char*)errorBlob->GetBufferPointer() << std::endl;
            errorBlob->Release();
        }
        return nullptr;
    }
    ID3D11ComputeShader* shader = nullptr;
    device->CreateComputeShader(shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(), nullptr, &shader);
    shaderBlob->Release();
    return shader;
}

IMFSourceReader* InitWebcam(UINT32& outWidth, UINT32& outHeight) {
    IMFAttributes* config = nullptr;
    MFCreateAttributes(&config, 1);
    config->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);

    IMFActivate** devices = nullptr;
    UINT32 count = 0;
    MFEnumDeviceSources(config, &devices, &count);
    
    if (count == 0) {
        std::cerr << "No webcam found!" << std::endl;
        if(config) config->Release();
        return nullptr;
    }
    
    IMFMediaSource* source = nullptr;
    devices[0]->ActivateObject(IID_PPV_ARGS(&source));
    
    for (UINT32 i = 0; i < count; i++) devices[i]->Release();
    CoTaskMemFree(devices);
    config->Release();
    
    if (!source) return nullptr;

    IMFAttributes* attributes = nullptr;
    MFCreateAttributes(&attributes, 1);
    attributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, 1);
    
    IMFSourceReader* reader = nullptr;
    MFCreateSourceReaderFromMediaSource(source, attributes, &reader);
    source->Release();
    attributes->Release();

    IMFMediaType* type = nullptr;
    MFCreateMediaType(&type);
    type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
    HRESULT hr = reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, NULL, type);
    type->Release();

    if (FAILED(hr)) {
        std::cerr << "Failed to set RGB32 format on webcam." << std::endl;
        reader->Release();
        return nullptr;
    }

    IMFMediaType* currentType = nullptr;
    reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, &currentType);
    MFGetAttributeSize(currentType, MF_MT_FRAME_SIZE, &outWidth, &outHeight);
    currentType->Release();

    return reader;
}

struct BlurParams {
    int radius;
    float sigma;
    float dirX;
    float dirY;
    uint32_t width;
    uint32_t height;
    float pad[2];
};

int main() {
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
    MFStartup(MF_VERSION);

    UINT32 width = 0, height = 0;
    std::cout << "Initializing Webcam..." << std::endl;
    IMFSourceReader* reader = InitWebcam(width, height);
    if (!reader) {
        MFShutdown();
        CoUninitialize();
        return -1;
    }
    std::cout << "Webcam initialized at " << width << "x" << height << std::endl;

    WNDCLASS wc = { 0 };
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = "WebcamBlur";
    RegisterClass(&wc);

    RECT rect = { 0, 0, (LONG)width, (LONG)height };
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindow("WebcamBlur", "DirectCompute GPU Webcam Blur", 
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 
        rect.right - rect.left, rect.bottom - rect.top, 
        nullptr, nullptr, wc.hInstance, nullptr);
    ShowWindow(hwnd, SW_SHOW);

    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    IDXGISwapChain* swapChain = nullptr;

    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount = 1;
    sd.BufferDesc.Width = width;
    sd.BufferDesc.Height = height;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_UNORDERED_ACCESS;
    sd.OutputWindow = hwnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;

    HRESULT hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &sd, &swapChain, &device, nullptr, &context);
    if (FAILED(hr)) {
        std::cerr << "Failed to create D3D11 device and swap chain!" << std::endl;
        return -1;
    }

    ID3D11ComputeShader* blurShader = CompileComputeShader(device, L"blur.hlsl", "CSMain");
    if (!blurShader) return -1;

    // Webcam Input Texture
    D3D11_TEXTURE2D_DESC camDesc = {};
    camDesc.Width = width;
    camDesc.Height = height;
    camDesc.MipLevels = 1;
    camDesc.ArraySize = 1;
    camDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    camDesc.SampleDesc.Count = 1;
    camDesc.Usage = D3D11_USAGE_DYNAMIC;
    camDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    camDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    ID3D11Texture2D* camTex = nullptr;
    device->CreateTexture2D(&camDesc, nullptr, &camTex);

    ID3D11ShaderResourceView* camSrv = nullptr;
    device->CreateShaderResourceView(camTex, nullptr, &camSrv);

    // Intermediate Texture for Separable Blur
    D3D11_TEXTURE2D_DESC tempDesc = {};
    tempDesc.Width = width;
    tempDesc.Height = height;
    tempDesc.MipLevels = 1;
    tempDesc.ArraySize = 1;
    tempDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    tempDesc.SampleDesc.Count = 1;
    tempDesc.Usage = D3D11_USAGE_DEFAULT;
    tempDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    ID3D11Texture2D* tempTex = nullptr;
    device->CreateTexture2D(&tempDesc, nullptr, &tempTex);

    ID3D11ShaderResourceView* tempSrv = nullptr;
    device->CreateShaderResourceView(tempTex, nullptr, &tempSrv);

    ID3D11UnorderedAccessView* tempUav = nullptr;
    device->CreateUnorderedAccessView(tempTex, nullptr, &tempUav);

    // Backbuffer UAV
    ID3D11Texture2D* backBuffer = nullptr;
    swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer));
    ID3D11UnorderedAccessView* bbUav = nullptr;
    device->CreateUnorderedAccessView(backBuffer, nullptr, &bbUav);
    backBuffer->Release();

    // Constant Buffer
    D3D11_BUFFER_DESC cbDesc = {};
    cbDesc.ByteWidth = sizeof(BlurParams);
    cbDesc.Usage = D3D11_USAGE_DEFAULT;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    ID3D11Buffer* cbBlur = nullptr;
    device->CreateBuffer(&cbDesc, nullptr, &cbBlur);

    std::cout << "\n======================================" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << " [UP]   Increase Blur Size & Strength" << std::endl;
    std::cout << " [DOWN] Decrease Blur Size & Strength" << std::endl;
    std::cout << " [ESC]  Exit Application" << std::endl;
    std::cout << "======================================\n" << std::endl;

    auto lastTime = std::chrono::high_resolution_clock::now();
    int frames = 0;

    while (g_running) {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (!g_running) break;

        IMFSample* sample = nullptr;
        DWORD streamIndex, flags;
        LONGLONG llTimeStamp;
        hr = reader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, &streamIndex, &flags, &llTimeStamp, &sample);
        
        if (SUCCEEDED(hr) && sample) {
            IMFMediaBuffer* buffer = nullptr;
            sample->ConvertToContiguousBuffer(&buffer);
            BYTE* data = nullptr;
            buffer->Lock(&data, nullptr, nullptr);
            
            D3D11_MAPPED_SUBRESOURCE mapped;
            context->Map(camTex, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
            
            LONG pitch = width * 4;
            // Direct copy (assuming the format is already top-down, no vertical flip needed)
            for(uint32_t y = 0; y < height; ++y) {
                memcpy((BYTE*)mapped.pData + y * mapped.RowPitch, data + y * pitch, pitch);
            }
            
            context->Unmap(camTex, 0);
            buffer->Unlock();
            buffer->Release();
            sample->Release();

            // FPS Calculation
            frames++;
            auto currentTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = currentTime - lastTime;
            if (elapsed.count() >= 1.0) {
                double fps = frames / elapsed.count();
                char title[256];
                sprintf_s(title, "DirectCompute GPU Webcam Blur - FPS: %.1f | Radius: %d | Sigma: %.1f", fps, g_radius, g_sigma);
                SetWindowTextA(hwnd, title);
                frames = 0;
                lastTime = currentTime;
            }
        }

        // Setup compute pipeline
        context->CSSetShader(blurShader, nullptr, 0);
        context->CSSetConstantBuffers(0, 1, &cbBlur);

        int dispatchX = (width + 15) / 16;
        int dispatchY = (height + 15) / 16;

        // -- Pass 1: Horizontal Blur (Cam -> Temp) --
        BlurParams params = { g_radius, g_sigma, 1.0f, 0.0f, width, height, {0,0} };
        context->UpdateSubresource(cbBlur, 0, nullptr, &params, 0, 0);
        
        ID3D11ShaderResourceView* srvH[] = { camSrv };
        context->CSSetShaderResources(0, 1, srvH);
        ID3D11UnorderedAccessView* uavH[] = { tempUav };
        context->CSSetUnorderedAccessViews(0, 1, uavH, nullptr);
        
        context->Dispatch(dispatchX, dispatchY, 1);

        // Unbind UAV/SRV to avoid read/write pipeline conflicts
        ID3D11UnorderedAccessView* nullUav[] = { nullptr };
        ID3D11ShaderResourceView* nullSrv[] = { nullptr };
        context->CSSetUnorderedAccessViews(0, 1, nullUav, nullptr);
        context->CSSetShaderResources(0, 1, nullSrv);

        // -- Pass 2: Vertical Blur (Temp -> Backbuffer) --
        params.dirX = 0.0f; params.dirY = 1.0f;
        context->UpdateSubresource(cbBlur, 0, nullptr, &params, 0, 0);
        
        ID3D11ShaderResourceView* srvV[] = { tempSrv };
        context->CSSetShaderResources(0, 1, srvV);
        ID3D11UnorderedAccessView* uavV[] = { bbUav };
        context->CSSetUnorderedAccessViews(0, 1, uavV, nullptr);
        
        context->Dispatch(dispatchX, dispatchY, 1);

        // Unbind
        context->CSSetUnorderedAccessViews(0, 1, nullUav, nullptr);
        context->CSSetShaderResources(0, 1, nullSrv);

        // Present to Window
        swapChain->Present(1, 0);
    }

    // Cleanup
    cbBlur->Release(); bbUav->Release();
    tempUav->Release(); tempSrv->Release(); tempTex->Release();
    camSrv->Release(); camTex->Release(); blurShader->Release();
    swapChain->Release(); context->Release(); device->Release();
    reader->Release();

    MFShutdown();
    CoUninitialize();
    return 0;
}
