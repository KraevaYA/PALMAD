#include "timer.cuh"

void start_timer(cudaEvent_t* start)
{
    cudaEventCreate(start);
    cudaEventRecord(*start);
}


float stop_timer(cudaEvent_t* start, cudaEvent_t* stop)
{
    float milliseconds = 0;
    
    cudaEventCreate(stop);
    cudaEventRecord(*stop);
    
    cudaEventSynchronize(*stop);
    cudaEventElapsedTime(&milliseconds, *start, *stop);
    
    return milliseconds;
}
