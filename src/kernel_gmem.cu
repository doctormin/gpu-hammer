#include "cuda_runtime.h"

// Should be multiple of 32
#define ARRAY_SIZE      268435456   // 256M
// Should be multiple of 32 and power of 2
#define NUM_THREADS     256
//
#define UNROLL_DEPTH    128

static __device__ char ld_arr[ARRAY_SIZE];

static __global__ void gmem_ld_hammer_kernel()
{
    constexpr int ntmo = NUM_THREADS - 1;
    int nd = ARRAY_SIZE >> 3;
    volatile double *ptr = (volatile double *)ld_arr;
    double x = 0;
    int idx = threadIdx.x + blockIdx.x * NUM_THREADS;
    int sz = NUM_THREADS * gridDim.x;
    int usz = sz * UNROLL_DEPTH;
    nd -= (nd % usz);
    for (int it = 0; it < 1000; ++it) {
        for (int i = idx; i < nd; i += usz) {
            #pragma unroll
            for (int j = 0; j < UNROLL_DEPTH; ++j) {
                x += ptr[i+j*sz];
            }
        }
    }
    // For avoiding compiler optimization.
    ((double *)ptr)[idx] = x;
}

extern "C" {

cudaError_t gmem_ld_hammer(cudaStream_t s, int nblks)
{
    dim3 grid = dim3(nblks, 1, 1);
    dim3 block = dim3(NUM_THREADS, 1, 1);
    return cudaLaunchKernel((void *)gmem_ld_hammer_kernel, grid, block, 0, 0, s);
}

} // extern "C"
