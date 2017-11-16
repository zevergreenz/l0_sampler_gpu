#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <fcntl.h>

#define BUFFER_SIZE 2
#define P 1272461 // a large prime number
#define Z 2       // a random number from [0,P-1]

// X terms produces an x-independent hash
#define NUMCOEFF 4

typedef struct
{
  unsigned int sum;
  unsigned int weight;
  unsigned int fingerprint;
} one_sparse_sampler;

typedef struct
{
  // Number of hash functions
  unsigned int k;

  // Number of one-sparse samplers in each row
  unsigned int s;

  // Number of coefficients in the hash function
  unsigned int numCoefficients;

  // coefficients
  unsigned int *coefficients;
  one_sparse_sampler *samplers;
} s_sparse_sampler;

unsigned int hash(unsigned int *coeff, unsigned int numCoefficients, unsigned int value) {
  unsigned int val = value;

  for (unsigned int i = 0; i < numCoefficients; i++) {
    val = (val * coeff[i] + val) % P;
  }
  return val;
}

__device__ unsigned int hash_gpu(unsigned int *coeff, unsigned int numCoefficients, unsigned int value) {
  unsigned int val = value;

  for (unsigned int i = 0; i < numCoefficients; i++) {
    val = (val * coeff[i] + val) % P;
  }
  return val;
}

void process(s_sparse_sampler sampler, unsigned int *buffer) {
  unsigned int index, update, hashVal;
  one_sparse_sampler *one_sampler;

  for (unsigned int j = 0; j < BUFFER_SIZE / 2; j++) {
    for (unsigned int i = 0; i < sampler.k; i++) {
      index   = buffer[j >> 1];
      update  = buffer[1 + (j >> 1)];
      hashVal = hash(&(sampler.coefficients[i * sampler.numCoefficients]),
                     sampler.numCoefficients,
                     index) % (2 * sampler.s);
      one_sampler               = &sampler.samplers[i * 2 * sampler.s + hashVal];
      one_sampler->weight      += update;
      one_sampler->sum         += index * update;
      one_sampler->fingerprint += (update * pow(Z, index));
    }
  }
}

__device__ unsigned int powMod(unsigned int z, unsigned int index) {
  unsigned int subpow;
  if(index == 0) {
    return 1;
  } else if(index & 1 == 1) {
    subpow = powMod(z, index >> 1);
    return (z * subpow * subpow) % P;
  } else {
    subpow = powMod(z, index >> 1);
    return (subpow * subpow) % P;
  }
}

__global__ void process_gpu(s_sparse_sampler sampler, unsigned int *buffer) {
  unsigned int i, index, update, hashVal;
  one_sparse_sampler *one_sampler;
  
  i = blockIdx.x;
  
  for (unsigned int j = 0; j < BUFFER_SIZE / 2; j++) {
    index  = buffer[j >> 1];
    update = buffer[1 + (j >> 1)];
    hashVal = hash_gpu( &(sampler.coefficients[i * sampler.numCoefficients]),
             sampler.numCoefficients,
             index) % (2 * sampler.s);
    one_sampler               = &sampler.samplers[i * 2 * sampler.s + hashVal];
    one_sampler->weight      += update;
    one_sampler->sum         += index * update;
    // This is slow! pow() only exists for single and double-precision floats
    // And we need a double to cover the range of unsigned integers
    one_sampler->fingerprint += (update * powMod(Z, index));
  }
}

/**
   This method returns an array containing all non-zero indices in the s_sparse
      sampler.
   As of now, this method may contain duplicated indices.
   TODO: Remove duplicates.
 */
unsigned int* query(s_sparse_sampler sampler, unsigned int& size) {
  unsigned int *result = (unsigned int *)malloc(2 * sampler.s * sampler.k * sizeof(unsigned int));

  size = 0;
  one_sparse_sampler *one_sampler;

  for (unsigned int i = 0; i < sampler.k; i++) {
    for (unsigned int j = 0; j < sampler.s * 2; j++) {
      one_sampler = &sampler.samplers[i * 2 * sampler.s + j];

      if (one_sampler->weight != 0) {
        unsigned int index = one_sampler->sum / one_sampler->weight;
        unsigned int error = one_sampler->fingerprint -
                    ((one_sampler->weight * pow(Z, index)));

        if ((unsigned int)error == 0) result[size++] = index;
      }
    }
  }
  return result;
}

void initialize_s_sparse_sampler(s_sparse_sampler *sampler,
                                 unsigned int               s,
                                 unsigned int               k,
                                 unsigned int               n) {
  sampler->k               = k;
  sampler->s               = s;
  sampler->numCoefficients = n;
  cudaMallocManaged((void **)&(sampler->samplers),     sizeof(unsigned int) * k * s * 2);
  cudaMallocManaged((void **)&(sampler->coefficients), sizeof(unsigned int) * k * n);

  for (unsigned int i = 0; i < k * n; i++) {
    sampler->coefficients[i] = rand();
  }
}

void sample(char *filename, unsigned int s, unsigned int k) {
  s_sparse_sampler sampler;
  unsigned int *buffer;

  initialize_s_sparse_sampler(&sampler, s, k, NUMCOEFF);
  cudaMallocManaged((void **)&buffer, sizeof(unsigned int) * BUFFER_SIZE);

  // Read data from file
  FILE *fdIn = fopen(filename, "r");

  while (!feof(fdIn)) {
    fscanf(fdIn, "%u %u", &buffer[0], &buffer[1]);

    for (int i = 0; i < BUFFER_SIZE / 2; i++) {
      // Sequencial processing
      process(sampler, buffer);

      // Parallel processing
      // process_gpu<<<sampler.k, 1>>>(sampler, buffer);
      // cudaDeviceSynchronize();
    }
  }

  // Synchronize parallel blocks.

  // Query the s-sparse sampler and print out
  unsigned int  size   = 0;
  unsigned int *result = query(sampler, size);

  for (unsigned int i = 0; i < size; i++) printf("%u ", result[i]);
  printf("\n");

  // Clean up
  cudaFree(buffer);

  // cudaFree(d_samplers);
}

int main(void) {
  sample("data_stream.txt", 15, 15);
  return 0;
}
