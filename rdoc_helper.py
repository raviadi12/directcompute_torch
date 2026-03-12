import ctypes
import sys

class RENDERDOC_API_1_1_2(ctypes.Structure):
    _fields_ = [
        ("GetAPIVersion", ctypes.c_void_p), ("SetCaptureOptionU32", ctypes.c_void_p),
        ("SetCaptureOptionF32", ctypes.c_void_p), ("GetCaptureOptionU32", ctypes.c_void_p),
        ("GetCaptureOptionF32", ctypes.c_void_p), ("SetFocusToggleKeys", ctypes.c_void_p),
        ("SetCaptureKeys", ctypes.c_void_p), ("GetOverlayBits", ctypes.c_void_p),
        ("MaskOverlayBits", ctypes.c_void_p), ("RemoveHooks", ctypes.c_void_p),
        ("UnloadCrashHandler", ctypes.c_void_p), ("SetCaptureFilePathTemplate", ctypes.c_void_p),
        ("GetCaptureFilePathTemplate", ctypes.c_void_p), ("GetNumCaptures", ctypes.c_void_p),
        ("GetCapture", ctypes.c_void_p), ("TriggerCapture", ctypes.c_void_p),
        ("IsTargetControlConnected", ctypes.c_void_p), ("LaunchReplayUI", ctypes.c_void_p),
        ("SetActiveWindow", ctypes.c_void_p),
        ("StartFrameCapture", ctypes.WINFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)),
        ("IsFrameCapturing", ctypes.c_void_p),
        ("EndFrameCapture", ctypes.WINFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p)),
    ]

def get_rdoc_api():
    print("\n[RDoc] Searching process memory for injected RenderDoc...")
    
    # 1. Access the core Windows memory manager
    kernel32 = ctypes.WinDLL('kernel32')
    kernel32.GetModuleHandleW.restype = ctypes.c_void_p
    kernel32.GetProcAddress.restype = ctypes.c_void_p
    kernel32.GetProcAddress.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    # 2. Look for the DLL that RenderDoc injected into our RAM
    rdoc_handle = kernel32.GetModuleHandleW("renderdoc.dll")
    
    if not rdoc_handle:
        print("[RDoc] ❌ FAIL: renderdoc.dll is NOT in memory.")
        print("[RDoc] 👉 Make sure 'Capture Child Processes' is checked in RenderDoc.")
        print("[RDoc] 👉 Make sure you are using a normal Python environment (not Windows Store).")
        return None
        
    print(f"[RDoc] ✅ Found injected renderdoc.dll at memory address: {hex(rdoc_handle)}")
    
    # 3. Get the exact memory address of the GetAPI function
    get_api_addr = kernel32.GetProcAddress(rdoc_handle, b"RENDERDOC_GetAPI")
    
    if not get_api_addr:
        print("[RDoc] ❌ FAIL: Could not find RENDERDOC_GetAPI function.")
        return None
        
    # 4. Create a callable Python function directly from that memory address!
    RENDERDOC_GetAPI = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))(get_api_addr)
    
    api_pointer = ctypes.c_void_p()
    
    # 5. Request API v1.1.2 (10012)
    res = RENDERDOC_GetAPI(10102, ctypes.byref(api_pointer))
    
    if res == 1 and api_pointer:
        print("[RDoc] 🚀 SUCCESS! API hooked via memory address. Ready to capture.")
        return ctypes.cast(api_pointer, ctypes.POINTER(RENDERDOC_API_1_1_2)).contents
    else:
        print(f"[RDoc] ❌ FAIL: GetAPI returned {res}.")
        
    return None