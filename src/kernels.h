#ifndef KERNELS_H_
#define KERNELS_H_

#include "cuda_runtime.h"

extern "C" cudaError_t fp32_hammer(cudaStream_t s, int nblks);
extern "C" cudaError_t l2_ld_hammer(cudaStream_t s, int nblks);
extern "C" cudaError_t gmem_ld_hammer(cudaStream_t s, int nblks);
extern "C" cudaError_t fp_hammer(cudaStream_t s, int nblks);
extern "C" cudaError_t gmem_fp_hammer(cudaStream_t s, int nblks);

extern "C" cudaError_t tensor_f16f16f16_hammer(cudaStream_t s, int nblks);
extern "C" cudaError_t tensor_f16f16f32_hammer(cudaStream_t s, int nblks);
extern "C" cudaError_t tensor_bf16bf16f32_hammer(cudaStream_t s, int nblks);
extern "C" cudaError_t tensor_tf32tf32f32_hammer(cudaStream_t s, int nblks);
extern "C" cudaError_t tensor_f64f64f64_hammer(cudaStream_t s, int nblks);
extern "C" cudaError_t tensor_s8s8s32_hammer(cudaStream_t s, int nblks);
extern "C" cudaError_t tensor_s4s4s32_hammer(cudaStream_t s, int nblks);
extern "C" cudaError_t tensor_b1b1s32_hammer(cudaStream_t s, int nblks);

#endif // KERNELS_H_
