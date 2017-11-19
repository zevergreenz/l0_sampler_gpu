#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

int main(int argc, char *argv[]){
  struct cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, 0);
  printf("using %i multiprocessors\n max threads per processor: %i \n"
    ,properties.multiProcessorCount
    ,properties.maxThreadsPerMultiProcessor);
  return 0;
}
