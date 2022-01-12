#include "cuda_runtime.h"
#include "cuda_fp16.h"

static __device__ float tensor_out;

static __global__ void tensor_hammer_kernel()
{
    __shared__ unsigned int smem[3];
    __half2 *A = (__half2 *)&smem[0];
    A[0] = __float2half2_rn(0.01);
    A[1] = __float2half2_rn(0.01);

    __half2 *B = (__half2 *)&smem[2];
    B[0] = __float2half2_rn(0.01);

    float C0[4] = { 0.01, 0.01, 0.01, 0.01 };
    float C1[4] = { 0.01, 0.01, 0.01, 0.01 };

    __syncthreads();

    for (int it = 0; it < 102400; ++it) {
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            asm (
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                "   { %0, %1, %2, %3 }, "
                "   { %4, %5 },         "
                "   { %6 },             "
                "   { %0, %1, %2, %3 }; "
                : "+f"(C0[0]), "+f"(C0[1]), "+f"(C0[2]), "+f"(C0[3])
                : "r"(smem[0]), "r"(smem[1]),
                  "r"(smem[2])
            );
            asm (
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                "   { %0, %1, %2, %3 }, "
                "   { %4, %5 },         "
                "   { %6 },             "
                "   { %0, %1, %2, %3 }; "
                : "+f"(C1[0]), "+f"(C1[1]), "+f"(C1[2]), "+f"(C1[3])
                : "r"(smem[0]), "r"(smem[1]),
                  "r"(smem[2])
            );
        }
    }
    if (C0[0] > 0.2) {
        tensor_out =
            C0[0] + C0[1] + C0[2] + C0[3] +
            C1[0] + C1[1] + C1[2] + C1[3];
    }
}

extern "C" {

cudaError_t tensor_hammer(cudaStream_t s, int nblks)
{
    dim3 grid = dim3(nblks, 1, 1);
    dim3 block = dim3(256, 1, 1);
    return cudaLaunchKernel((void *)tensor_hammer_kernel, grid, block, 0, 0, s);
}

} // extern "C"
