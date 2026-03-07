import ctypes
import os
import numpy as np
import atexit

# Load the DLL
dll_path = os.path.join(os.path.dirname(__file__), 'directcompute.dll')
try:
    _lib = ctypes.CDLL(dll_path)
except OSError as e:
    raise RuntimeError(f"Could not load directcompute.dll from {dll_path}: {e}")

_lib.InitDirectCompute.argtypes = [ctypes.c_wchar_p]
_lib.InitDirectCompute.restype = ctypes.c_bool
_lib.CleanupDirectCompute.restype = None
_lib.MatMul.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]
_lib.MatMul.restype = ctypes.c_bool

# Initialize on module load
shader_path = os.path.join(os.path.dirname(__file__), 'matmul_coarsened_2d.hlsl')
if not _lib.InitDirectCompute(shader_path):
    raise RuntimeError(f"Failed to initialize DirectCompute or compile shader at {shader_path}.")

def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Performs hardware-accelerated matrix multiplication using DirectCompute (2D Coarsening).
    """
    if A.dtype != np.float32 or B.dtype != np.float32:
        A = A.astype(np.float32)
        B = B.astype(np.float32)
    
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Matrix dimensions do not align: {A.shape} and {B.shape}")
    
    M, K = A.shape
    _, N = B.shape
    
    # Ensure memory is contiguous C-style array
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    C = np.zeros((M, N), dtype=np.float32)
    
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    success = _lib.MatMul(A_ptr, B_ptr, C_ptr, M, K, N)
    if not success:
        raise RuntimeError("DirectCompute MatMul execution failed.")
        
    return C

@atexit.register
def _cleanup():
    _lib.CleanupDirectCompute()
