#pragma once

#include <cuda_device_runtime_api.h>

void start_timer(cudaEvent_t* start);
float stop_timer(cudaEvent_t* start, cudaEvent_t* stop);





































































// #ifndef __KERNELS_CUH__
// #define __KERNELS_CUH__

// __global__ void dot_product_kernel(float *x, float *y, float *dot, unsigned int n);

// #endif
