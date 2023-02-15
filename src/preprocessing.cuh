#pragma once

#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>

#include "parameters.hpp"

template<typename TYPE>
__global__ void gpu_compute_statistics(TYPE *d_T, TYPE *d_mean, TYPE *d_std, unsigned int N, unsigned int m)
{
    unsigned int tid = threadIdx.x;
    unsigned int thread_idx = (blockIdx.x*blockDim.x)+tid;

    TYPE mean, std;
    while (thread_idx < N)
    {
        mean = 0;
        std = 0;

        for (int i = 0; i < m; i++)
        {
	        mean += d_T[thread_idx+i];
	        std += d_T[thread_idx+i]*d_T[thread_idx+i];
        }

	    std = std/m;
	    mean = mean/m;

        d_mean[thread_idx] = mean;
        d_std[thread_idx] = sqrt(std - mean*mean);

        thread_idx += blockDim.x*gridDim.x;
    }
}


template<typename TYPE>
__global__ void gpu_update_statistics(TYPE *d_T, TYPE *d_mean, TYPE *d_std, unsigned int N, unsigned int m)
{
    unsigned int tid = threadIdx.x;
    unsigned int thread_idx = (blockIdx.x*blockDim.x)+tid;

    TYPE mean, std;

    while (thread_idx < N)
    {
        mean = 0;
        std = 0;

        mean = (((TYPE)(m-1)/m)*d_mean[thread_idx] + d_T[thread_idx+m-1]/m);
        std = sqrt(((TYPE)(m-1)/m)*(d_std[thread_idx]*d_std[thread_idx]+(d_mean[thread_idx]-d_T[thread_idx+m-1])*(d_mean[thread_idx]-d_T[thread_idx+m-1])/m));

        thread_idx += blockDim.x*gridDim.x;
    }
}


template<typename TYPE>
void cuda_compute_statistics(TYPE *d_T, TYPE *d_mean, TYPE *d_std, unsigned int N, unsigned int m)
{
    unsigned int num_blocks = (int)ceil(N/(float)BLOCK_SIZE);
    dim3 blockDim = dim3(BLOCK_SIZE, 1, 1);
    dim3 gridDim = dim3(num_blocks, 1, 1);
    gpu_compute_statistics<TYPE><<<gridDim, blockDim>>>(d_T, d_mean, d_std, N, m);
}


template<typename TYPE>
void cuda_update_statistics(TYPE *d_T, TYPE *d_mean, TYPE *d_std, unsigned int N, unsigned int m)
{
    unsigned int num_blocks = (int)ceil(N/(float)BLOCK_SIZE);
    dim3 blockDim = dim3(BLOCK_SIZE, 1, 1);
    dim3 gridDim = dim3(num_blocks, 1, 1);
    gpu_update_statistics<TYPE><<<gridDim, blockDim>>>(d_T, d_mean, d_std, N, m);
}
