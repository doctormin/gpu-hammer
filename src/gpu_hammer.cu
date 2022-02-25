#include <cstdio>
#include <cstdlib>
#include "cuda_runtime.h"

#include "kernels.h"
#include "common.h"


int main(int argc, const char *argv[])
{
    int num_sm;
    CUDACHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));

    // CUDACHECK(fp32_hammer(cudaStreamDefault, 108));
    // CUDACHECK(l2_ld_hammer(cudaStreamDefault, 108));
    // CUDACHECK(gmem_ld_hammer(cudaStreamDefault, 108));
    // CUDACHECK(fp_hammer(cudaStreamDefault, 108));
    CUDACHECK(gmem_fp_hammer(cudaStreamDefault, 108));

    // CUDACHECK(tensor_f16f16f16_hammer(cudaStreamDefault, 108));
    // CUDACHECK(tensor_f16f16f32_hammer(cudaStreamDefault, 108));
    // CUDACHECK(tensor_bf16bf16f32_hammer(cudaStreamDefault, 108));
    // CUDACHECK(tensor_tf32tf32f32_hammer(cudaStreamDefault, 108));
    // CUDACHECK(tensor_f64f64f64_hammer(cudaStreamDefault, 108));
    // CUDACHECK(tensor_s8s8s32_hammer(cudaStreamDefault, 108));
    // CUDACHECK(tensor_s4s4s32_hammer(cudaStreamDefault, 108));
    // CUDACHECK(tensor_b1b1s32_hammer(cudaStreamDefault, 108));

    CUDACHECK(cudaDeviceSynchronize());

    cudaDeviceReset();
    return 0;
}
