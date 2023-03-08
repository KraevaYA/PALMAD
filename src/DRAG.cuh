/*+++++++++++++++++++++++++++++++++
Project: PALMAD (Parallel Arbitrary Length MERLIN-based Anomaly Discovery)
Source file: DRAG.cuh
Purpose: Parallel implementation of the DRAG algorithm in CUDA
Author(s): Yana Kraeva (kraevaya@susu.ru)
+++++++++++++++++++++++++++++++++*/

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <assert.h>
#include <cublas_v2.h>
#include <vector>
#include <algorithm>
#include <bits/stdc++.h>

#include "DRAG_kernels.cuh"
#include "parameters.hpp"
#include "timer.cuh"

using namespace std;

template <typename TYPE>
void do_DRAG(TYPE *d_T, TYPE *d_mean, TYPE *d_std, int *d_cand, int *d_neighbor, TYPE *d_nnDist, unsigned int N, unsigned int m, unsigned int w, TYPE r, unsigned int max_N, TYPE *range_discord_profile)
{

    int D_size = 0;
    int C_size = 0;

    int *h_cand_gpu = (int *)malloc(N*sizeof(int));
    TYPE *h_nnDist_gpu = (TYPE *)malloc(N*sizeof(TYPE));

    unsigned int num_blocks;
    dim3 blockDim, gridDim; 
        
    // some events to count the execution time
    cudaEvent_t start, stop;
    float phase_time = 0;

    // Phase 1. Candidate Selection Algorithm
    num_blocks = ceil(N/(float)SEGMENT_N);
    unsigned int N_pad = num_blocks*SEGMENT_N + m - 1;

    blockDim = dim3(SEGMENT_LEN, 1, 1);
    gridDim = dim3(num_blocks, 1, 1);

    start_timer(&start);
    gpu_candidate_select<TYPE><<<gridDim, blockDim, (2*SEGMENT_N+2*m-2)*sizeof(TYPE)>>>(d_T, d_mean, d_std, d_cand, d_neighbor, d_nnDist, N_pad, r, m);
    phase_time = stop_timer(&start, &stop);

    num_blocks = ceil(N_pad/(float)BLOCK_SIZE);
    blockDim = dim3(BLOCK_SIZE, 1, 1);
    gridDim = dim3(num_blocks, 1, 1);

    gpu_define_candidates<<<gridDim, blockDim>>>(d_cand, d_neighbor, N_pad);

    cudaMemcpy(h_cand_gpu, d_cand, N*sizeof(int), cudaMemcpyDeviceToHost);
    C_size = 0;
    for (int i = 0; i < N; i++)
    {
	    C_size += h_cand_gpu[i];
    }

    // Phase 2. Discord Refinement Algorithm
    num_blocks = ceil((N_pad-m+1)/(float)SEGMENT_N);
    N_pad = num_blocks*SEGMENT_N;

    blockDim = dim3(SEGMENT_N, 1, 1);
    gridDim = dim3(num_blocks, 1, 1);

    start_timer(&start);
    gpu_discords_refine<TYPE><<<gridDim, blockDim, (2*SEGMENT_N+2*m-2)*sizeof(TYPE)>>>(d_T, d_mean, d_std, d_cand, d_nnDist, N_pad, m, r);
    phase_time = stop_timer(&start, &stop);
    
    cudaMemcpy(h_cand_gpu, d_cand, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_nnDist_gpu, d_nnDist, N*sizeof(TYPE), cudaMemcpyDeviceToHost);

    TYPE nnDist = 0;

    for (int i = 0; i < N; i++)
    {
	    nnDist = round(h_nnDist_gpu[i] * 10000.0) / 10000.0;
	    if (h_nnDist_gpu[i] == 1) {
	        range_discord_profile[i] = nnDist;
	        D_size++;
	    }
	    else
	        range_discord_profile[i] = -1;	
    }

    for (int i = N; i < max_N; i++)
	    range_discord_profile[i] = -1;
}
