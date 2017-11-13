#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <fcntl.h>

#define BUFFER_SIZE 2
#define P 1272461 // a large prime number
#define Z 2   // a random number from [0,P-1]

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
  } return val;
}

void process(s_sparse_sampler sampler, int *buffer) {
  // int row   = blockIdx.x;
  // int col   = *buffer % (2 * (*s));
  // int index = row * 2 * (*s) + col;
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

int query(s_sparse_sampler sampler) {
  int index;
  one_sparse_sampler *one_sampler;

  for (int i = 0; i < sampler.k; i++) {
    for (int j = 0; j < sampler.s * 2; j++) {
      one_sampler = &sampler.samplers[i * 2 * sampler.s + j];
      if (one_sampler->weight != 0) {
        index = one_sampler->sum / one_sampler->weight;
        int error = one_sampler->fingerprint - ((one_sampler->weight * pow(Z, index)));
        if ((int) error == 0) {
          return index;
        }
      }
    }
  }
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
      process(sampler, buffer);
    }
  }

  // Query the s-sparse sampler
  printf("query: %d\n", query(sampler));

  // Clean up
  cudaFree(buffer);

  // cudaFree(d_samplers);
}

int main(void) {
  sample("data_stream.txt", 1, 1);
  return 0;
}
