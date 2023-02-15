#pragma once

#include <math.h>
#include <stdio.h>
#include <float.h>

#include "parameters.hpp"

template <typename TYPE>
__global__ void gpu_candidate_select(TYPE *d_T, TYPE *d_mean, TYPE *d_std, int *d_cand, int *d_neighbor, TYPE *d_nnDist, unsigned int N, TYPE r, unsigned int m)
{
    unsigned int tid = threadIdx.x; 
    unsigned int blockSize = blockDim.x;
    unsigned int segment_ind = blockIdx.x*blockSize;
    unsigned int chunk_ind = segment_ind + m - 1;
    TYPE nnDist = FLT_MAX;
    TYPE min_nnDist = FLT_MAX;
    bool non_overlap;

    extern __shared__ TYPE dynamicMem[];
    TYPE *segment = dynamicMem;
    TYPE *chunk = (TYPE*)&segment[SEGMENT_N+m-1];

    __shared__ int cand[SEGMENT_N];
    __shared__ TYPE dot_col[SEGMENT_N];
    __shared__ TYPE dot_row[SEGMENT_N];
    __shared__ TYPE dot_inter[SEGMENT_N];
    __shared__ int all_rej[1];

    cand[tid] = 1;

    if (tid == 0)
	    all_rej[0] = 1;

    int ind = tid;
    int segment_len = SEGMENT_N+m-1;

    while (ind < segment_len)
    {
	    segment[ind] = d_T[segment_ind+ind];
	    chunk[ind] = d_T[chunk_ind+ind];
	    ind += blockSize;
    }

    dot_col[tid] = 0;
    dot_row[tid] = 0;

    __syncthreads();

    // calculate dot for the first column and row (the first chunk)
    for (int j = 0; j < m; j++)
    {
        dot_col[tid] += segment[j]*chunk[j+tid];
        dot_row[tid] += segment[j+tid]*chunk[j];
    }

    __syncthreads();

    nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind])/(m*d_std[segment_ind+tid]*d_std[chunk_ind]));

    if (isinf(abs(nnDist)) || (nnDist < 0) || (nnDist >= FLT_MAX))
	    nnDist = 2.0*m;

    non_overlap = (abs((int)(segment_ind+tid-chunk_ind)) < (m-1)) ? 0 : 1;

    if (non_overlap) 
    {
        if (nnDist < r)
        {
            cand[tid] = 0;
            atomicMin(d_neighbor+chunk_ind, 0);
        }
        else
	        min_nnDist = min(min_nnDist, nnDist);
    }

    // calculate dot for rows from second to last (the first chunk)
    for (int j = 0; j < blockSize-1; j++)
    {
	    if (tid > 0)
	        dot_inter[tid] = dot_row[tid-1];

        __syncthreads();

        if (tid > 0)
            dot_row[tid] = dot_inter[tid] + segment[m+tid-1]*chunk[m+j] - segment[tid-1]*chunk[j];

        __syncthreads();

        if (tid == 0)
            dot_row[tid] = dot_col[j+1];

        nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind+j+1])/(m*d_std[segment_ind+tid]*d_std[chunk_ind+j+1]));
        
        if (isinf(abs(nnDist)) || (nnDist < 0) || (nnDist >= FLT_MAX))
	        nnDist = 2.0*m;

	    non_overlap = (abs((int)(segment_ind+tid-chunk_ind-j-1)) < (m-1)) ? 0 : 1;

        if (non_overlap) 
	    {
            if (nnDist < r)
            {
                cand[tid] = 0;
                atomicMin(d_neighbor+chunk_ind+j+1, 0);
            }
            else
		        min_nnDist = min(min_nnDist, nnDist);
        }
    }

    __syncthreads();

    if (tid == 0)
    {
        all_rej[0] = 0;
        for (int k = 0; k < blockSize; k++)
            if (cand[k] == 1)
	        {
	            all_rej[0] = cand[k];
		        break;
	        }
    }

    __syncthreads();

    chunk_ind += blockSize;

    // process chunks from the second to last
    while ((chunk_ind < N) && (all_rej[0] != 0))
    {
        dot_col[tid] = 0;
	    ind = tid;

	    while (ind < segment_len)
	    {
	        chunk[ind] = d_T[chunk_ind+ind];
	        ind += blockSize;
        }
 	
	    __syncthreads();
	
        for (int j = 0; j < m; j++)
	    {
            dot_col[tid] += segment[j]*chunk[j+tid];
	    }

	    if (tid > 0)
	        dot_inter[tid] = dot_row[tid-1];

	    __syncthreads();
	
	    if (tid > 0)
            dot_row[tid] = dot_inter[tid] + segment[m+tid-1]*chunk[m-1] - segment[tid-1]*d_T[chunk_ind-1];
	    else
            dot_row[tid] = dot_col[0];

	    __syncthreads();

        nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind])/(m*d_std[segment_ind+tid]*d_std[chunk_ind]));

        if (isinf(abs(nnDist)) || (nnDist < 0) || (nnDist >= FLT_MAX))
            nnDist = 2.0*m;

        if (cand[tid] != 0)
        {
            if (nnDist < r)
            {
                cand[tid] = 0;
                atomicMin(d_neighbor+chunk_ind, 0);
            }
            else
               min_nnDist = min(min_nnDist, nnDist);
        }

        for (int j = 0; j < blockSize-1; j++)
        {
	        if (tid > 0)
                dot_inter[tid] = dot_row[tid-1];

            __syncthreads();

            if (tid > 0)
		        dot_row[tid] = dot_inter[tid] + segment[m+tid-1]*chunk[m+j] - segment[tid-1]*chunk[j];

            if (tid == 0)
                dot_row[tid] = dot_col[j+1];

            nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind+j+1])/(m*d_std[segment_ind+tid]*d_std[chunk_ind+j+1]));

            if (isinf(abs(nnDist)) || (nnDist < 0) || (nnDist >= FLT_MAX))
	            nnDist = 2.0*m;


            if (cand[tid] != 0)
            {
                if (nnDist < r)
                {
                    cand[tid] = 0;
                    atomicMin(d_neighbor+chunk_ind+j+1, 0);
                }
                else
                    min_nnDist = min(min_nnDist, nnDist);
            }
        }

        __syncthreads();

        if (tid == 0)
        {
            all_rej[0] = 0;
            for (int k = 0; k < blockSize; k++)
                if (cand[k] == 1)
                {
                    all_rej[0] = cand[k];
                    break;
                }
        }

        __syncthreads();

        chunk_ind += blockSize;

    }

    if (segment_ind+tid < N)
    {
        d_cand[segment_ind+tid] = cand[tid];
	    d_nnDist[segment_ind+tid] = min_nnDist;
    }
}


__global__ void gpu_define_candidates(int *d_cand, int *d_neighbor, unsigned int N)
{
    unsigned int tid = threadIdx.x; 
    unsigned int thread_id = blockIdx.x*blockDim.x+tid;

    if (thread_id < N)
    {
	    d_cand[thread_id] = d_cand[thread_id] * d_neighbor[thread_id];
	    thread_id += blockDim.x*gridDim.x;
    }

}


template <typename TYPE>
__global__ void gpu_discords_refine(TYPE *d_T, TYPE *d_mean, TYPE *d_std, int *d_cand, TYPE *d_nnDist, unsigned int N, int m, TYPE r)
{
    unsigned int tid = threadIdx.x; 
    unsigned int blockSize = blockDim.x;
    unsigned int segment_ind = blockIdx.x*blockSize;
    unsigned int chunk_ind = 0;
    TYPE nnDist = FLT_MAX;
    TYPE min_nnDist = d_nnDist[segment_ind+tid];
    bool non_overlap;
    int ind = tid;
    int step = 0;
    int segment_len = SEGMENT_N+m-1;

    extern __shared__ TYPE dynamicMem[];
    TYPE *segment = dynamicMem;
    TYPE *chunk = (TYPE*)&segment[SEGMENT_N+m-1];

    __shared__ int cand[SEGMENT_N];
    __shared__ TYPE dot_col[SEGMENT_N];
    __shared__ TYPE dot_row[SEGMENT_N];
    __shared__ TYPE dot_inter[SEGMENT_N];
    __shared__ int all_rej[1];

    cand[tid] = d_cand[segment_ind+tid];

    __syncthreads();

    if (tid == 0)
    {
        all_rej[0] = 0;
        for (int k = 0; k < blockSize; k++)
            if (cand[k] == 1)
            {
                all_rej[0] = cand[k];
                break;
            }
    }

    __syncthreads();

    if (all_rej[0] != 0) 
    {
        while (ind < segment_len)
        {
	        segment[ind] = d_T[segment_ind+ind];
	        ind += blockSize;
        }

	    while ((chunk_ind < segment_ind-blockSize) && (all_rej[0] != 0))
        {
	        dot_col[tid] = 0;
	    
	        ind = tid;
	    
	        while (ind < segment_len)
            {
	            chunk[ind] = d_T[chunk_ind+ind];
	            ind += blockSize;
            }

	        __syncthreads();

	        if (step == 0) {
	            dot_row[tid] = 0;

                for (int j = 0; j < m; j++)
                {
                    dot_col[tid] += segment[j]*chunk[j+tid];
                    dot_row[tid] += segment[j+tid]*chunk[j];
                }

	        }
            else
            {
                for (int j = 0; j < m; j++)
                    dot_col[tid] += segment[j]*chunk[j+tid];

                if (tid > 0)
                     dot_inter[tid] = dot_row[tid-1];

		        __syncthreads();
	
		        if (tid > 0)
            	    dot_row[tid] = dot_inter[tid] + segment[m+tid-1]*chunk[m-1] - segment[tid-1]*d_T[chunk_ind-1];
		        else
            	    dot_row[tid] = dot_col[0];
	        }

            __syncthreads();

            nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind])/(m*d_std[segment_ind+tid]*d_std[chunk_ind]));

            if (isinf(abs(nnDist)) || (nnDist < 0) || (nnDist >= FLT_MAX))
                nnDist = 2.0*m;

            if (cand[tid] != 0)
            {
                if (nnDist < r)
                {
                    cand[tid] = 0;
                    min_nnDist = -FLT_MAX;
                }
                else
                    min_nnDist = min(min_nnDist, nnDist);
            }

            for (int j = 0; j < blockSize-1; j++)
            {
                if (tid > 0)
                    dot_inter[tid] = dot_row[tid-1];

                __syncthreads();

                if (tid > 0)
                    dot_row[tid] = dot_inter[tid] + segment[m+tid-1]*chunk[m+j] - segment[tid-1]*chunk[j];

                __syncthreads();

                if (tid == 0)
                    dot_row[tid] = dot_col[j+1];

                nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind+j+1])/(m*d_std[segment_ind+tid]*d_std[chunk_ind+j+1]));

                if (isinf(abs(nnDist)) || (nnDist < 0) || (nnDist >= FLT_MAX))
                    nnDist = 2.0*m;

                if (cand[tid] != 0) {
                    if (nnDist < r)
                    {
                        cand[tid] = 0;
                        min_nnDist = -FLT_MAX;
                    }
                    else
                        min_nnDist = min(min_nnDist, nnDist);
                }
            }

            __syncthreads();

            if (tid == 0)
            {
                all_rej[0] = 0;
                for (int k = 0; k < blockSize; k++)
                    if (cand[k] == 1)
                    {
                        all_rej[0] = cand[k];
                        break;
                    }
            }

            __syncthreads();

            chunk_ind += blockSize;
            step++;
        }

        while ((chunk_ind < segment_ind) && (all_rej[0] != 0))
        {
            dot_row[tid] = 0;
            dot_col[tid] = 0;
            ind = tid;

            while (ind < segment_len)
            {
                chunk[ind] = d_T[chunk_ind+ind];
                ind += blockSize;
            }

            __syncthreads();

            for (int j = 0; j < m; j++)
            {
                dot_col[tid] += segment[j]*chunk[j+tid];
                dot_row[tid] += segment[j+tid]*chunk[j];
            }

            __syncthreads();

            nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind])/(m*d_std[segment_ind+tid]*d_std[chunk_ind]));

            if (isinf(abs(nnDist)) || (nnDist < 0) || (nnDist >= FLT_MAX))
                nnDist = 2.0*m;

            non_overlap = (abs((int)(segment_ind+tid-chunk_ind)) < (m-1)) ? 0 : 1;

            if ((non_overlap) && (cand[tid] != 0))
            {
                if (nnDist < r)
                {
                    cand[tid] = 0;
                    min_nnDist = -FLT_MAX;
                }
                else
                    min_nnDist = min(min_nnDist, nnDist);
            }

            for (int j = 0; j < blockSize-1; j++)
            {
                if (tid > 0)
                    dot_inter[tid] = dot_row[tid-1];

                __syncthreads();

                if (tid > 0)
                    dot_row[tid] = dot_inter[tid] + segment[m+tid-1]*chunk[m+j] - segment[tid-1]*chunk[j];

                __syncthreads();

                if (tid == 0)
                    dot_row[tid] = dot_col[j+1];

                nnDist = 2*m*(1-(dot_row[tid]-m*d_mean[segment_ind+tid]*d_mean[chunk_ind+j+1])/(m*d_std[segment_ind+tid]*d_std[chunk_ind+j+1]));

                if (isinf(abs(nnDist)) || (nnDist < 0) || (nnDist >= FLT_MAX))
                    nnDist = 2.0*m;

                non_overlap = (abs((int)(segment_ind+tid-chunk_ind-j-1)) < (m-1)) ? 0 : 1;

                if ((non_overlap) && (cand[tid] != 0))
                {
                    if (ED_dist < r)
                    {
                        cand[tid] = 0;
                        min_nnDist = -FLT_MAX;
                    }
                    else
                        min_nnDist = min(min_nnDist, nnDist);
                }
            }

            __syncthreads();

            if (tid == 0)
            {
                all_rej[0] = 0;
                for (int k = 0; k < blockSize; k++)
                    if (cand[k] == 1)
                    {
                        all_rej[0] = cand[k];
                        break;
                    }
            }

            __syncthreads();

            chunk_ind += blockSize;
        }

        if (segment_ind+tid < N)
        {
            d_cand[segment_ind+tid] *= cand[tid];
            d_nnDist[segment_ind+tid] = min_nnDist;
        }
    }
} // end void
