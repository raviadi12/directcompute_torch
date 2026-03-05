import numpy as np
import time
from directthon import directcompute

def test_matmul():
    SIZE = 1024
    print(f"Testing DirectCompute Python Bridge (Size: {SIZE}x{SIZE})")
    
    # Generate random matrices
    A = np.random.rand(SIZE, SIZE).astype(np.float32)
    B = np.random.rand(SIZE, SIZE).astype(np.float32)
    
    print("Running Numpy (CPU) MatMul...")
    start_cpu = time.perf_counter()
    C_ref = np.dot(A, B)
    end_cpu = time.perf_counter()
    print(f"Numpy CPU Time: {(end_cpu - start_cpu) * 1000:.2f} ms")
    
    print("Running DirectCompute (GPU) MatMul...")
    # Warmup
    _ = directcompute.matmul(A, B)
    
    start_gpu = time.perf_counter()
    C_test = directcompute.matmul(A, B)
    end_gpu = time.perf_counter()
    print(f"DirectCompute GPU Time: {(end_gpu - start_gpu) * 1000:.2f} ms")
    
    print(f"Speedup vs Numpy: {(end_cpu - start_cpu) / (end_gpu - start_gpu):.2f}x")
    
    # Verify correctness
    max_error = np.max(np.abs(C_ref - C_test))
    print(f"Max Error vs Numpy: {max_error}")
    
    if max_error < 1e-2:
        print("Result: VERIFIED SUCCESSFULLY!")
    else:
        print("Result: VERIFICATION FAILED!")

if __name__ == "__main__":
    test_matmul()
