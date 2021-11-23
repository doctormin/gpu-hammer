#include "cuda_runtime.h"

static __device__ float fma_out;

static __global__ void fp32_hammer_kernel()
{
    float a = -1e5f;
    float b = -1e5f;
    for (int it = 0; it < 102400; ++it) {
        #pragma unroll
        for (int i = 0; i < 256; ++i) {
            a = __fmaf_ru(a, 0.9999f, 0.01f);
            b = __fmaf_ru(b, 0.9999f, 0.01f);
        }
    }
    if (a < 0) {
        // This is for avoiding compiler optimization.
        // Program never reach here.
        fma_out = a + b;
    }
}


extern "C" {

cudaError_t fp32_hammer(cudaStream_t s, int nblks)
{
    dim3 grid = dim3(nblks, 1, 1);
    dim3 block = dim3(256, 1, 1);
    return cudaLaunchKernel((void *)fp32_hammer_kernel, grid, block, 0, 0, s);
}

} // extern "C"
