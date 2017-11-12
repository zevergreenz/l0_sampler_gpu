#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <fcntl.h>

#define BUFFER_SIZE 2
#define P 1272461               // a large prime number
#define Z 32909                 // a random number from [0,P-1]

//****************************************************************
// This code may not compile, to be tested on CUDA capable devices
//****************************************************************

typedef struct
{
    long long sum = 0;
    long long weight = 0;
    long long fingerprint = 0;
} one_sparse_sampler;

__global__ void process(one_sparse_sampler *samplers, int *buffer, int *s, int *k) {
    int row = blockIdx.x;
    int col = *buffer % (2 * (*s));
    int index = row * 2 * (*s) + col;
    int x = buffer[0];              // index of the vector following lecture notes
    int a = buffer[1];              // either 1 or -1 following lecture notes
    samplers[index].weight += a;
    samplers[index].sum += x * a;
    samplers[index].fingerprint += (a * pow((double) Z, (double) x));
}

void s_sparse_sampler(int s, int k) {
    int array_size = 2 * s * k;
    int sampler_size = sizeof(one_sparse_sampler);

    int *d_s, *d_k;
    int *d_buffer;                      // device copy of the data
    one_sparse_sampler *samplers;       // host copy of the samplers
    one_sparse_sampler *d_samplers;     // device copy of the samplers
    int *buffer;                        // host copy of the data

    // Allocate memory
    cudaMalloc((void**)&d_s, sizeof(int));
    cudaMalloc((void**)&d_k, sizeof(int));
    samplers = (one_sparse_sampler *) malloc(array_size * sampler_size);
    cudaMalloc((void **)&d_samplers, array_size * sampler_size);
    buffer = (int *) malloc(BUFFER_SIZE * sizeof(int));
    cudaMalloc((void **)&d_buffer, BUFFER_SIZE * sizeof(int));

    // Copy memory from host to device
    cudaMemcpy(d_s, &s, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);

    // Read data from file
    FILE *fdIn = fopen("10int.dat", "r");
    while ( fgets((char*) buffer, BUFFER_SIZE * sizeof(int), fdIn )) {
        // Copy data from host to device
        cudaMemcpy(d_buffer, buffer, BUFFER_SIZE * sizeof(int), cudaMemcpyHostToDevice);

        for (int i = 0; i < BUFFER_SIZE; i++) {
            // Launch the process kernel on GPU
            process<<<k,1>>>(d_samplers, d_buffer, d_s, d_k);
        }

    }

    // Copy result back to host
    cudaMemcpy(samplers, d_samplers, array_size * sampler_size, cudaMemcpyDeviceToHost);

    // Clean up
    free(samplers);
    cudaFree(d_samplers);
}

int main(void) {

    return 0;
}
