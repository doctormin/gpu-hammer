#include "cuda_runtime.h"

// Should be multiple of 32
#define L2_CACHE_SIZE   6291456   // 6M
// Should be multiple of 32 and power of 2
#define NUM_THREADS     256

static __device__ char arr[L2_CACHE_SIZE];

// Referred to code in https://arxiv.org/pdf/1804.06826.pdf
static __global__ void l2_ld_hammer_kernel()
{
    constexpr int ntmo = NUM_THREADS - 1;
    constexpr int nd = L2_CACHE_SIZE / 8;
    double x = 0;
    int tid = threadIdx.x;
    for (int it = 0; it < 500; ++it) {
        double *ptr = (double *)arr;
        for (int i = 0; i < nd; i += NUM_THREADS) {
            #pragma unroll
            for (int j = 0; j < NUM_THREADS; j += 32) {
                int offset = (tid + j) & ntmo;
                asm volatile ("{\t\n"
                    ".reg .f64 val;\n\t"
                    "ld.global.cg.f64 val, [%1];\n\t"
                    "add.f64 %0, val, %0;\n\t"
                    "}" : "+d"(x) : "l"(ptr+offset) : "memory"
                );
            }
            ptr += 32;
        }
    }
    // For avoiding compiler optimization.
    ((double *)arr)[tid] = x;
}

extern "C" {

cudaError_t l2_ld_hammer(cudaStream_t s, int nblks)
{
    dim3 grid = dim3(nblks, 1, 1);
    dim3 block = dim3(NUM_THREADS, 1, 1);
    return cudaLaunchKernel((void *)l2_ld_hammer_kernel, grid, block, 0, 0, s);
}

} // extern "C"
