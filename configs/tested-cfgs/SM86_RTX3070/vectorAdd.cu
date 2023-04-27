
/* Copyright (c) . All rights reserved. */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(int *data1, int *data2, int *X , int *Y , int *Z, int numElements) {

int N = 2, i = 0;

do {
    int t1 = threadIdx.x;
    int t2 = t1 + i; //add.s32 %r2, %r5, %r1;
    int t3 = data1[t2];  //A
    int t4 = 0;  //

    if (t3 != t4) {  //setp.ne.s32	%p1, %r6, 0; split warp
        int t5 = data2[t2]; //B
        if (t5 != t4) {
            X[t2] +=1;  //C
        } else {
            Y[t2] +=2;  //D
        }
        Z[t2] += X[t2]; //E

    } else {
        Z[t2] += 3; //F
    }
    Z[t2] += X[t2];//G
    i++;
    } while (i<N);
}

int main(void) {
  cudaError_t err = cudaSuccess;

  int numElements = 32;
  size_t size = numElements * sizeof(int);

  int *h_data1_in  = (int *)malloc(size);
  int *h_data2_in  = (int *)malloc(size);
  int *h_data1_out = (int *)malloc(size);
  int *h_data2_out = (int *)malloc(size);
  int *h_data3_out = (int *)malloc(size);

  for (int i = 0; i < numElements; ++i) {
    h_data1_in[i] = i;
    h_data2_in[i] = numElements - i;
  }

  int *d_data1_in = NULL;
  err = cudaMalloc((void **)&d_data1_in, size);
  
  int *d_data2_in = NULL;
  err = cudaMalloc((void **)&d_data2_in, size);

  int *d_data1_out = NULL;
  err = cudaMalloc((void **)&d_data1_out, size);
  int *d_data2_out = NULL;
  err = cudaMalloc((void **)&d_data2_out, size);
  int *d_data3_out = NULL;
  err = cudaMalloc((void **)&d_data3_out, size);

  printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_data1_in,  h_data1_in, size, cudaMemcpyHostToDevice);
  err = cudaMemcpy(d_data2_in,  h_data2_in, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 32;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_data1_in, d_data2_in, d_data1_out,d_data2_out , d_data3_out,numElements);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  printf("Copy output data from CUDA device to the host memory\n");
  err = cudaMemcpy(h_data1_out, d_data1_out, size, cudaMemcpyDeviceToHost);
  err = cudaMemcpy(h_data2_out, d_data2_out, size, cudaMemcpyDeviceToHost);
  err = cudaMemcpy(h_data3_out, d_data3_out, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < numElements; i++) {
    printf("%d ", h_data1_out[i]);
    printf("%d ", h_data2_out[i]);
    printf("%d ", h_data3_out[i]);
    }
  printf("\n");

  err = cudaFree(d_data1_in);
  err = cudaFree(d_data2_in);
  err = cudaFree(d_data1_out);
  err = cudaFree(d_data2_out);
  err = cudaFree(d_data3_out);

  free(h_data1_in);
  free(h_data2_in);
  free(h_data1_out);
  free(h_data2_out);
  free(h_data3_out);

  printf("Done\n");
  return 0;
}

