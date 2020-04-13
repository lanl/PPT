#include <stdio.h>
#include <cuda.h>

#define ITERATIONS 20000
#define N 100000


// Device code
__global__ void VecAdd(const int* A, const int* B, int* C, int size)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  for(int n = 0 ; n < 100; n++) {
      C[i] += A[i] + B[i];
  }
}

static void
compute(int iters)
{
  size_t size = N * sizeof(int);
  int threadsPerBlock = 0;
  int blocksPerGrid = 0;
  int i;
  int *d_A, *d_B, *d_C;

  // Allocate vectors in device memory
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);


  // Invoke kernel (multiple times to make sure we have time for
  // sampling)
  threadsPerBlock = 1024;
  blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  for (i = 0; i < iters; i++) {
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
  }
  cudaDeviceSynchronize();
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

int
main(int argc, char *argv[]){
  
    // run kernel while sampling
    compute(ITERATIONS);
}
