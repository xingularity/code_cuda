#ifndef _KERNEL_CU_
#define _KERNEL_CU_

__global__ void testKernel(double* g_idata, double* g_odata){
    extern __shared__ double sdata[];

    const unsigned int bid = blockIdx.x;
    const unsigned int tid_in_block = threadIdx.x;
    const unsigned int tid_in_grid = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid_in_block] = g_idata[tid_in_grid];
    __syncthreads();

    sdata[tid_in_block] *= bid;
    __syncthreads();

    g_odata[tid_in_grid] = sdata[tid_in_block];
}

#endif
