#ifndef KERNELS_H_
#define KERNELS_H_

#include "cuda_runtime.h"

extern "C" cudaError_t fp32_hammer(cudaStream_t s, int nblks);
extern "C" cudaError_t l2_ld_hammer(cudaStream_t s, int nblks);
extern "C" cudaError_t gmem_ld_hammer(cudaStream_t s, int nblks);

#endif // KERNELS_H_
