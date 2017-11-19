#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <fcntl.h>

#define BUFFER_SIZE 64
#define P 179426239 // a large prime number
#define Z 2       // a random number from [0,P-1]

// X terms produces an x-independent hash
#define NUMCOEFF 4

#define K 15

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

typedef struct {
  unsigned int n;
  s_sparse_sampler *samplers;
} l0_sampler;

long long wall_clock_time()
{
#ifdef __linux__
	struct timespec tp;
	clock_gettime(CLOCK_REALTIME, &tp);
	return (long long)(tp.tv_nsec + (long long)tp.tv_sec * 1000000000ll);
#else
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (long long)(tv.tv_usec * 1000 + (long long)tv.tv_sec * 1000000000ll);
#endif
}

void checkCudaError(cudaError_t rc)
{
  if (rc != cudaSuccess)
    printf("Last CUDA error %s\n", cudaGetErrorString(rc));
}
void checkCudaError(){
  cudaError_t rc = cudaGetLastError();
  checkCudaError(rc);
}
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

void process_s_sparse(s_sparse_sampler sampler, unsigned int *buffer) {
  unsigned int index, update, hashVal;
  one_sparse_sampler *one_sampler;

  for (unsigned int j = 0; j < BUFFER_SIZE; j+= 2) {
    for (unsigned int i = 0; i < sampler.k; i++) {
      index   = buffer[j];
      update  = buffer[j + 1];
      hashVal = hash(&(sampler.coefficients[i * sampler.numCoefficients]),
                     sampler.numCoefficients,
                     index) % (2 * sampler.s);
      if(index)
        // printf("Updating Sampler (%i, %i) with value %u %u\n", i, hashVal, index, update);
      one_sampler               = &sampler.samplers[i * 2 * sampler.s + hashVal];
      one_sampler->weight      += update;
      one_sampler->sum         += index * update;
      one_sampler->fingerprint += (update * pow(Z, index));
      one_sampler->fingerprint %= P;
    }
  }
}
void process(l0_sampler sampler, unsigned int *buffer) {
  for(int i = 0; i < sampler.n; i++) {
    process_s_sparse(sampler.samplers[i], buffer);
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

__global__ void process_gpu(l0_sampler l0_sampler, unsigned int *buffer) {
  unsigned int i, j, index, update, hashVal, m, s, k;
  s_sparse_sampler *sampler;
  one_sparse_sampler *one_sampler;
  i = threadIdx.x;
  j = threadIdx.y;
  k = blockIdx.x;

  sampler = &(l0_sampler.samplers[j]);

  m = sampler->numCoefficients;
  s = sampler->s;

  index = buffer[k * 2];
  update = buffer[1 + (k * 2)];
  hashVal = hash_gpu( &(sampler->coefficients[i * m]),
           m,
           index) % (2 * sampler->s);
  one_sampler               = &sampler->samplers[i * 2 * s + hashVal];
  // printf("I'm at %i\n", i);
  atomicAdd(&(one_sampler->weight), update);
  atomicAdd(&(one_sampler->sum), index *update);
  // This is slow! pow() only exists for single and double-precision floats
  // And we need a double to cover the range of unsigned integers
  atomicAdd(&(one_sampler->fingerprint), (update * powMod(Z, index) % P));
}

/**
   This method returns an array containing all non-zero indices in the s_sparse
      sampler.
   As of now, this method may contain duplicated indices.
   TODO: Remove duplicates.
 */
unsigned int* query(s_sparse_sampler sampler, unsigned int& size) {
  unsigned int *result = (unsigned int *)malloc(2 * sampler.s * sampler.k * sizeof(unsigned int));

  cudaDeviceSynchronize();
  size = 0;
  one_sparse_sampler *one_sampler;

  for (int i = 0; i < sampler.k; i++) {
    for (int j = 0; j < sampler.s * 2; j++) {
      one_sampler = &(sampler.samplers[i * 2 * sampler.s + j]);
      // printf("sampler (%i, %i) has weight %u\n", i, j, one_sampler->weight);
      if (one_sampler->weight != 0) {
        unsigned int index = one_sampler->sum / one_sampler->weight;
        unsigned int error = one_sampler->fingerprint -
                    ((one_sampler->weight * pow(Z, index)));
        if (error % P == 0) result[size++] = index;
      }
    }
  }
  return result;
}

void initialize_s_sparse_sampler(s_sparse_sampler *sampler,
                                 unsigned int               s,
                                 unsigned int               k,
                                 unsigned int               n) {
  int i = 0;
  sampler->k               = k;
  sampler->s               = s;
  sampler->numCoefficients = n;
  cudaMallocManaged((void **)&(sampler->samplers),     sizeof(one_sparse_sampler) * k * s * 2);
  cudaMallocManaged((void **)&(sampler->coefficients), sizeof(unsigned int) * k * n);

  for(i = 0; i < k * s * 2; i++) {
    sampler->samplers[i].sum = 0;
    sampler->samplers[i].weight = 0;
    sampler->samplers[i].fingerprint = 0;
  }
  for (i = 0; i < k * n; i++)
  {
    sampler->coefficients[i] = rand();
  }
}
void initialize_l0_sampler(l0_sampler *sampler, int s, int k, int m, int n) {
  int i = 0;
  sampler->n = n;
  cudaMallocManaged((void **)&(sampler->samplers), sizeof(s_sparse_sampler) * n);
  for (i = 0; i < n; i++) {
    initialize_s_sparse_sampler(&(sampler->samplers[i]), s, k, m);
  }
}

void sample(char *filename, unsigned int s, unsigned int k, unsigned int n) {
  l0_sampler seq_sampler, gpu_sampler;
  
  unsigned int *buffer, *buffer2;
  cudaError_t rc;

  long long start_time;
  float seq_time = 0, gpu_time = 0;

  initialize_l0_sampler(&seq_sampler, s, k, NUMCOEFF, n);
  initialize_l0_sampler(&gpu_sampler, s, k, NUMCOEFF, n);

  rc = cudaMallocManaged((void **)&buffer, sizeof(unsigned int) * BUFFER_SIZE);
  checkCudaError(rc);

  int i;


  // First evaluate sequential program, then parallel program
  // Read data from file
  FILE *fdIn = fopen(filename, "r");
  start_time = wall_clock_time();
  while (!feof(fdIn))
  {
    for(i = 0; i < BUFFER_SIZE; i+= 2) {
      if(feof(fdIn)) {
        buffer[i] = 0;
        buffer[i + 1] = 0;
      } else {
        fscanf(fdIn, "%u %u", &buffer[i], &buffer[i + 1]);
      }
    }
    process(seq_sampler, buffer);

  }
  seq_time = (float)(wall_clock_time() - start_time) / 1000000000;
  fclose(fdIn);

  unsigned int size;
  unsigned int *result;
  printf("Sequential Vector: Time %1.2f s\n", seq_time);
  // Query the s-sparse sampler and print out
  size = 0;
  // result = query(seq_sampler, size);

  // for (i = 0; i < size; i++)
  //   printf("%u ", result[i]);
  // printf("\n");


  // Read data from file
  fdIn = fopen(filename, "r");
  dim3 blocks(k, n);
  int flip_flop = 0;
  unsigned int *readBuffer = buffer;
  start_time = wall_clock_time();
  while (!feof(fdIn))
  {
    cudaDeviceSynchronize();
    // Basically, process one buffer while reading in the next one
    // if(flip_flop) {
    //   readBuffer = buffer;
    // } else {
    //   readBuffer = buffer2;
    // }
    for (i = 0; i < BUFFER_SIZE; i+=2)
    {
      if (feof(fdIn))
      {
        buffer[i] = 0;
        buffer[i + 1] = 0;
      }
      else
      {
        // printf("Reading values for: %i,%i to %#10X \n", i, i + 1, &readBuffer[i]);
        fscanf(fdIn, "%u %u", &buffer[i], &buffer[i + 1]);
        // printf("Read values for:%i : %i, %i\n", i, readBuffer[i], readBuffer[i + 1]);
      }
    }
    // Synchronize parallel blocks.
    process_gpu<<<BUFFER_SIZE / 2, blocks>>>(gpu_sampler, buffer);
    checkCudaError();
    flip_flop = 1 - flip_flop;
  }
  cudaDeviceSynchronize();
  gpu_time = (float)(wall_clock_time() - start_time) / 1000000000;


  printf("GPU Vector: Time %1.2f s\n", gpu_time);
  // Query the s-sparse sampler and print out
  // size   = 0;
  // result = query(gpu_sampler, size);

  // for (i = 0; i < size; i++) printf("%u ", result[i]);
  // printf("\n");

  // Clean up
  cudaFree(buffer);
  // cudaFree(buffer2);

  // cudaFree(d_samplers);
}

int main(int argc, char **argv) {
  // Reads datafile from args
  if(argc < 2){
    printf("Usage: %s <input_file>", argv[0]);
    return 0;
  }
  sample(argv[1], 25, 50, 20);
  return 0;
}
