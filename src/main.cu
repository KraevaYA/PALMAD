#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <assert.h>
#include <cublas_v2.h>
#include <cuda_device_runtime_api.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <bits/stdc++.h>

#include "DRAG.cuh"
#include "preprocessing.cuh"
#include "DRAG_types.hpp"
#include "parameters.hpp"
#include "IOdata.hpp"
#include "timer.cuh"


using namespace std;

int define_subsequence_number_with_pad(unsigned int n, unsigned int min_m, unsigned int max_m)
{
    unsigned int pad_N = 0;
    unsigned int pad = ceil((n-min_m+1)/(float)SEGMENT_N)*SEGMENT_N + 2*min_m - 2 - n;
    unsigned int pad_per_segment = ceil(pad/SEGMENT_N);
    unsigned int delta_m = pad_per_segment*SEGMENT_N - pad - 2;
    unsigned int m_N_max = delta_m + min_m;

    if (a == 1)
        m_N_max = m_N_max + min_m;

    if (m_N_max > max_m)
        m_N_max = max_m;

    pad_N = ceil((n-m_N_max+1)/(float)SEGMENT_N)*SEGMENT_N + 2*m_N_max - 2 - m_N_max + 1;

    return pad_N;
}


bool sortPairs(const pair<DATA_TYPE, int> &x, const pair<DATA_TYPE, int> &y)
{
    return x.first > y.first;
}


int find_discord_count(DATA_TYPE *range_discord_profile, unsigned int N)
{

    int D_size = 0;

    for (int i = 0; i < N; i++)
    {
	    if (range_discord_profile[i] != -1)
	        D_size++;
    }

    return D_size;
}


void find_non_overlap_discords(DATA_TYPE *range_discord_profile, vector<pair<DATA_TYPE, int>>&top_k_discords, unsigned int N, unsigned int m)
{

    vector<pair<DATA_TYPE, int>>discords;
    int D_size = 0;

    for (int i = 0; i < N; i++)
    {
        if (range_discord_profile[i] != -1)
        {
            discords.push_back(make_pair(range_discord_profile[i], i));
            D_size = D_size + 1;
        }
    }
    sort(discords.begin(), discords.end(), sortPairs);

    if (discords.size() > 0)
    {
        top_k_discords.push_back(make_pair((DATA_TYPE)discords[0].first, (int)discords[0].second));
        int non_overlap = 1;

        for (int i = 1; i < D_size; i++)
        {
            non_overlap = 1;

            for (int j = 0; j < top_k_discords.size(); j++)
            {
                if (abs((int)(top_k_discords[j].second - discords[i].second)) < m)
                {
                    non_overlap = 0;
                    break;
                }
            }
                if (non_overlap)
                    top_k_discords.push_back(make_pair(discords[i].first, discords[i].second));
        }
    }
}


int main(int argc, char *argv[])
{
    char *file_name = argv[1];          // name of input file with time series
    unsigned int n = atoi(argv[2]);     // length of time series
    unsigned int min_m = atoi(argv[3]); // min length of subsequence
    unsigned int max_m = atoi(argv[4]); // max length of subsequence
    unsigned int K = atoi(argv[5]);     // top k discords of each length
    char *profile_file_name = argv[6];  // name of output file with time series
    char *discord_file_name = argv[7];  // name of output file with time series

    DATA_TYPE r = 2*sqrt(min_m);
    unsigned int w = min_m; // window
    unsigned int N = n - min_m + 1;
    unsigned int num_lengths = max_m - min_m + 1;
    unsigned int m;
    DATA_TYPE M, S;
    unsigned int D_size_non_overlap = 0;

    unsigned int N_pad = 0;
    unsigned int n_pad = 0;

    N_pad = define_subsequence_number_with_pad(n, min_m, max_m);
    n_pad = N_pad + max_m;

    // some events to count the execution time
    cudaEvent_t start, stop;
    float step_time = 0, length_time = 0;

    // Allocate memory space on host
    DATA_TYPE *h_T = (DATA_TYPE *)malloc(n_pad*sizeof(DATA_TYPE));
    int *h_cand = (int *)malloc(N_pad*sizeof(int));
    int *h_neighbor = (int *)malloc(N_pad*sizeof(int));
    DATA_TYPE *h_nnDist = (DATA_TYPE *)malloc(N_pad*sizeof(DATA_TYPE));
    DATA_TYPE *range_discord_profile = (DATA_TYPE *)malloc((n-min_m+1)*sizeof(DATA_TYPE));

    Discord *top1_discords = (Discord *)malloc(num_lengths*sizeof(Discord));
    for (int i = 0; i < num_lengths; i++)
        top1_discords[i] = {-1, -FLT_MAX};
    vector<pair<DATA_TYPE, int>>top_k_discords;

    // Allocate memory space on the device
    DATA_TYPE *d_T, *d_mean, *d_std, *d_nnDist;
    int *d_cand, *d_neighbor;

    cudaMalloc((void **) &d_T, n_pad*sizeof(DATA_TYPE));
    cudaMalloc((void **) &d_mean, N_pad*sizeof(DATA_TYPE));
    cudaMalloc((void **) &d_std, N_pad*sizeof(DATA_TYPE));
    cudaMalloc((void **) &d_cand, N_pad*sizeof(int));
    cudaMalloc((void **) &d_neighbor, N_pad*sizeof(int));
    cudaMalloc((void **) &d_nnDist, N_pad*sizeof(DATA_TYPE));

    for (int i = 0; i < N_pad; i++)
    {
        h_nnDist[i] = FLT_MAX;
        h_cand[i] = 1;
	    h_neighbor[i] = 1;
    }

    read_ts<DATA_TYPE>(file_name, h_T, n_pad);
    
    // copy time series from host to device memory
    cudaMemcpy(d_T, h_T, n_pad*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    
    printf("m;phase1_time;#Candidates;phase2_time;#Discords;all_time\n");

    // Init mean and std for subsequences of minimum length
    start_timer(&start);
    cuda_compute_statistics<DATA_TYPE>(d_T, d_mean, d_std, N_pad, min_m);
    step_time = stop_timer(&start, &stop);
    length_time = step_time;

    while (top1_discords[0].dist < 0)
    {
        // copy input data to device
    	cudaMemcpy(d_cand, h_cand, N_pad*sizeof(int), cudaMemcpyHostToDevice);
	    cudaMemcpy(d_neighbor, h_neighbor, N_pad*sizeof(int), cudaMemcpyHostToDevice);
    	cudaMemcpy(d_nnDist, h_nnDist, N_pad*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    	start_timer(&start);
        do_DRAG<DATA_TYPE>(d_T, d_mean, d_std, d_cand, d_neighbor, d_nnDist, N, min_m, w, r*r, n-min_m+1, range_discord_profile);
        step_time = stop_timer(&start, &stop);
	    length_time += step_time;

	    find_non_overlap_discords(range_discord_profile, top_k_discords, N, min_m);
	    D_size_non_overlap = top_k_discords.size();

        if (D_size_non_overlap >= K)
        {
            if (top_k_discords[0].first != FLT_MAX)
                top1_discords[0].dist = top_k_discords[0].first;
            else
                top1_discords[0].dist = 2*min_m;
            top1_discords[0].ind = top_k_discords[0].second;
            write_range_discord_profile<DATA_TYPE>(profile_file_name, range_discord_profile, n-min_m+1, min_m);
            write_discords<DATA_TYPE>(discord_file_name, top_k_discords, K, min_m);
        }

	    length_time = 0;
        r = r * 0.5;
	    top_k_discords.clear();
    }

    for (int i = 1; i < 5; i++)
    {
	    if (i == num_lengths)
	        break;
        m = min_m + i;
        w = m;
        N = n - m + 1;
	    length_time = 0;
	    D_size_non_overlap = 0;

	    // Update mean and std for subsequences of the current length
        start_timer(&start);
        cuda_update_statistics<DATA_TYPE>(d_T, d_mean, d_std, N_pad, m);
        step_time = stop_timer(&start, &stop);
	    length_time = step_time;

        r = sqrt(top1_discords[i-1].dist)*0.99;

        while (top1_discords[i].dist < 0)
        {
            // copy input data to device
    	    cudaMemcpy(d_cand, h_cand, N_pad*sizeof(int), cudaMemcpyHostToDevice);
	        cudaMemcpy(d_neighbor, h_neighbor, N_pad*sizeof(int), cudaMemcpyHostToDevice);
    	    cudaMemcpy(d_nnDist, h_nnDist, N_pad*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

            start_timer(&start);
            do_DRAG<DATA_TYPE>(d_T, d_mean, d_std, d_cand, d_neighbor, d_nnDist, N, m, w, r*r, n-min_m+1, range_discord_profile);
            step_time = stop_timer(&start, &stop);
	        length_time += step_time;
	    
            find_non_overlap_discords(range_discord_profile, top_k_discords, N, m);
            D_size_non_overlap = top_k_discords.size();

	        if (D_size_non_overlap >= K)
            {
		        if (top_k_discords[0].first != FLT_MAX)
	        	    top1_discords[i].dist = top_k_discords[0].first;
		        else
			        top1_discords[i].dist = 2*m;
	            top1_discords[i].ind = top_k_discords[0].second;
	            write_range_discord_profile<DATA_TYPE>(profile_file_name, range_discord_profile, n-min_m+1, m);
	            write_discords<DATA_TYPE>(discord_file_name, top_k_discords, K, m);
	        }

	        length_time = 0;
	        r = r*0.99;
	        top_k_discords.clear();
        }
    }
    
    if (num_lengths > 5)
        for (int i = 5; i < num_lengths; i++)
        {
            m = min_m + i;
            w = m;
            N = n - m + 1;
	        length_time = 0;
	        D_size_non_overlap = 0;

            // Update mean and std for subsequences of the current length
            start_timer(&start);
            cuda_update_statistics<DATA_TYPE>(d_T, d_mean, d_std, N_pad, m);
            step_time = stop_timer(&start, &stop);
	        length_time = step_time;

            M = 0;
            S = 0;
        
            for (int j = i-5; j < i; j++)
                M += sqrt(top1_discords[j].dist);
            M = M*0.2;
        
            for (int j = i-5; j < i; j++)
                S += pow((sqrt(top1_discords[j].dist) - M), 2);
        
            S = sqrt(S*0.2);
            r = M - 2*S;
        
            while (top1_discords[i].dist < 0)
            {
                // copy input data to device
    	        cudaMemcpy(d_cand, h_cand, N_pad*sizeof(int), cudaMemcpyHostToDevice);
	            cudaMemcpy(d_neighbor, h_neighbor, N_pad*sizeof(int), cudaMemcpyHostToDevice);
    	        cudaMemcpy(d_nnDist, h_nnDist, N_pad*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

                start_timer(&start);
                do_DRAG<DATA_TYPE>(d_T, d_mean, d_std, d_cand, d_neighbor, d_nnDist, N, m, w, r*r, n-min_m+1, range_discord_profile);
                step_time = stop_timer(&start, &stop);
		        length_time += step_time;

		        find_non_overlap_discords(range_discord_profile, top_k_discords, N, m);
	            D_size_non_overlap = top_k_discords.size();

	            if (D_size_non_overlap >= K)
                {
		            if (top_k_discords[0].first != FLT_MAX)
	            	    top1_discords[i].dist = top_k_discords[0].first;
		            else
			            top1_discords[i].dist = 2*m;
	                top1_discords[i].ind = top_k_discords[0].second;
	                write_range_discord_profile<DATA_TYPE>(profile_file_name, range_discord_profile, n-min_m+1, m);
	                write_discords<DATA_TYPE>(discord_file_name, top_k_discords, K, m);
	            }

		        length_time = 0;
                r = r - S;
		        top_k_discords.clear();
            }
        }

    for (int i = 0; i < num_lengths; i++)
        printf("The top-1 discord of length %d is at %d, with a discord distance of %.6lf\n", min_m+i, top1_discords[i].ind, sqrt(top1_discords[i].dist));

    // deallocate device memory
    cudaFree(d_T);
    cudaFree(d_mean);
    cudaFree(d_std);
    cudaFree(d_cand);
    cudaFree(d_neighbor);
    cudaFree(d_nnDist);

    // deallocate host memory
    free(h_T);
    free(h_cand);
    free(h_neighbor);
    free(h_nnDist);

    return 0;
}

