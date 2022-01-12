#include "cuda_runtime.h"
#include "cuda_fp16.h"

#define COMP_HALF   0

#if (COMP_HALF == 1)
static __device__ int hfma_out;
#endif // (COMP_HALF == 1)
static __device__ float sfma_out;
static __device__ double dfma_out;

static __global__ void fp_hammer_kernel()
{
#if (COMP_HALF == 1)
    const __half2 c1 = __float2half2_rn(0.9999);
    const __half2 c2 = __float2half2_rn(0.01);
    const int *pc1 = (int*)&c1;
    const int *pc2 = (int*)&c2;
    __half2 ha = __float2half2_rn(-1e3f);
    __half2 hb = __float2half2_rn(-1e3f);
    int *pha = (int*)&ha;
    int *phb = (int*)&hb;
#endif // (COMP_HALF == 1)
    float sa = -1e5f;
    float sb = -1e5f;
    double da = -1e7;
    double db = -1e7;
    __syncthreads();
    for (int it = 0; it < 102400; ++it) {
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            asm ("fma.rn.f64    %0, %0, 0.9999, 0.01;" : "+d"(da));
            asm ("fma.rn.f32    %0, %0, 0.9999, 0.01;" : "+f"(sa));
            asm ("fma.rn.f32    %0, %0, 0.9999, 0.01;" : "+f"(sb));
#if (COMP_HALF == 1)
            asm ("fma.rn.f16x2  %0, %0, %1, %2;" : "+r"(*pha) : "r"(*pc1), "r"(*pc2));
#endif // (COMP_HALF == 1)
            asm ("fma.rn.f64    %0, %0, 0.9999, 0.01;" : "+d"(db));
            asm ("fma.rn.f32    %0, %0, 0.9999, 0.01;" : "+f"(sa));
            asm ("fma.rn.f32    %0, %0, 0.9999, 0.01;" : "+f"(sb));
#if (COMP_HALF == 1)
            asm ("fma.rn.f16x2  %0, %0, %1, %2;" : "+r"(*phb) : "r"(*pc1), "r"(*pc2));
#endif // (COMP_HALF == 1)
        }
    }
    if (sa < 0) {
        // This is for avoiding compiler optimization.
        // Program never reach here.
        sfma_out = sa + sb;
        dfma_out = da + db;
#if (COMP_HALF == 1)
        hfma_out = *pha + *phb;
#endif // (COMP_HALF == 1)
    }
}

extern "C" {

cudaError_t fp_hammer(cudaStream_t s, int nblks)
{
    dim3 grid = dim3(nblks, 1, 1);
    dim3 block = dim3(512, 1, 1);
    return cudaLaunchKernel((void *)fp_hammer_kernel, grid, block, 0, 256, s);
}

} // extern "C"
