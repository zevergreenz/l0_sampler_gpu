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
  int sum;
  int weight;
  int fingerprint;
} one_sparse_sampler;

typedef struct
{
  // Number of hash functions
  int k;

  // Number of one-sparse samplers in each row
  int s;

  // Number of coefficients in the hash function
  int numCoefficients;

  // coefficients
  int                *coefficients;
  one_sparse_sampler *samplers;
} s_sparse_sampler;

int hash(int *coeff, int numCoefficients, int value) {
  int val = value;

  for (int i = 0; i < numCoefficients; i++) {
    val = (val * coeff[i] + val) % P;
  } 
  return (val < 0) ? val + P : val;
}

__device__ void hash_gpu(int &hashVal, int *coeff, int numCoefficients, int value) {
  int val = value;

  for (int i = 0; i < numCoefficients; i++) {
    val = (val * coeff[i] + val) % P;
  }
  hashVal = (val < 0) ? val + P : val;
}

void process(s_sparse_sampler sampler, int *buffer) {
  int index, update, hashVal;
  one_sparse_sampler *one_sampler;

  for (int j = 0; j < BUFFER_SIZE / 2; j++) {
    for (int i = 0; i < sampler.k; i++) {
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



__global__ void process_gpu(s_sparse_sampler sampler, int *buffer) {
  int i, index, update, hashVal;
  one_sparse_sampler *one_sampler;

  i = blockIdx.x;

  for (int j = 0; j < BUFFER_SIZE / 2; j++) {
    index  = buffer[j >> 1];
    update = buffer[1 + (j >> 1)];
    hash_gpu(hashVal,
             &(sampler.coefficients[i * sampler.numCoefficients]),
             sampler.numCoefficients,
             index);
    hashVal                  %= (2 * sampler.s);
    one_sampler               = &sampler.samplers[i * 2 * sampler.s + hashVal];
    one_sampler->weight      += update;
    one_sampler->sum         += index * update;
    // This is slow! pow() only exists for single and double-precision floats
    // And we need a double to cover the range of integers
    one_sampler->fingerprint += (update * pow((double)Z, (double)index));
  }
}

/**
   This method returns an array containing all non-zero indices in the s_sparse
      sampler.
   As of now, this method may contain duplicated indices.
   TODO: Remove duplicates.
 */
int* query(s_sparse_sampler sampler, int& size) {
  int *result = (int *)malloc(2 * sampler.s * sampler.k * sizeof(int));

  size = 0;
  one_sparse_sampler *one_sampler;

  for (int i = 0; i < sampler.k; i++) {
    for (int j = 0; j < sampler.s * 2; j++) {
      one_sampler = &sampler.samplers[i * 2 * sampler.s + j];

      if (one_sampler->weight != 0) {
        int index = one_sampler->sum / one_sampler->weight;
        int error = one_sampler->fingerprint -
                    ((one_sampler->weight * pow(Z, index)));

        if ((int)error == 0) result[size++] = index;
      }
    }
  }
  return result;
}

void initialize_s_sparse_sampler(s_sparse_sampler *sampler,
                                 int               s,
                                 int               k,
                                 int               n) {
  sampler->k               = k;
  sampler->s               = s;
  sampler->numCoefficients = n;
  cudaMallocManaged((void **)&(sampler->samplers),     sizeof(int) * k * s * 2);
  cudaMallocManaged((void **)&(sampler->coefficients), sizeof(int) * k * n);

  for (int i = 0; i < k * n; i++) {
    sampler->coefficients[i] = rand();
  }
}

void sample(char *filename, int s, int k) {
  s_sparse_sampler sampler;
  int *buffer;

  initialize_s_sparse_sampler(&sampler, s, k, NUMCOEFF);
  cudaMallocManaged((void **)&buffer, sizeof(int) * BUFFER_SIZE);

  // Read data from file
  FILE *fdIn = fopen(filename, "r");

  while (!feof(fdIn)) {
    fscanf(fdIn, "%d %d", &buffer[0], &buffer[1]);

    for (int i = 0; i < BUFFER_SIZE / 2; i++) {
      // Sequencial processing
      process(sampler, buffer);

      // Parallel processing
      // process_gpu<<<sampler.k, 1>>>(sampler, buffer);
    }
  }

  // Synchronize parallel blocks.
  cudaDeviceSynchronize();

  // Query the s-sparse sampler and print out
  int  size   = 0;
  int *result = query(sampler, size);

  for (int i = 0; i < size; i++) printf("%d ", result[i]);
  printf("\n");

  // Clean up
  cudaFree(buffer);

  // cudaFree(d_samplers);
}

int main(void) {
  sample("data_stream.txt", 15, 15);
  return 0;
}
