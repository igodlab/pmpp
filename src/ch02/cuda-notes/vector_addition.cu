#include <cstdio>
#include <chrono>

// Kernel function for adding arrays
__global__
void vecAddKernel(float *A, float *B, float *C, int n){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

// Compute vector addition sum C_h = A_h + B_h
void vecAdd_d(float* A_h, float* B_h, float* C_h, int n){
  // Part 1 - allocate memory for A, B, C
  int size = n * sizeof(float);
  float *A_d, *B_d, *C_d;
  cudaMalloc((void **)&A_d, size);
  cudaMalloc((void **)&B_d, size);
  cudaMalloc((void **)&C_d, size);

  // Copy arrays from Host to Device: A, C, B to global (device) memory
  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

  // Invoke kernel
  vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

  // Copy result from device to host
  cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

  // Free global memory
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

// Compute vector addition sum in CPU only C_h = A_h + B_h
void vecAdd_h(float* A_h, float* B_h, float* C_h, int n){
  for (int i = 0; i < n; i++) {
    C_h[i] = A_h[i] + B_h[i];
  }
}

void printVec(float* V, int n){
  printf("[");
  for (int i = 0; i < n; i++){
    printf("%f", V[i]);
    if (i < n - 1) printf(", ");
  }
  printf("]\n");
}

int main(void) {
  const int n = 10;
  float A[] = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5};
  float B[] = {2.25, 4.25, 6.25, 8.25, 10.25, 12.25, 14.25, 16.25, 18.25, 20.25};
  float C_h[n], C_d[n];

  // CPU
  auto start_h = std::chrono::high_resolution_clock::now();
  vecAdd_h(A, B, C_h, n);
  auto end_h = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_h = end_h - start_h;
  printf("CPU:\n----\n"); printVec(C_h, n);
  printf("vecAdd_h took %.4f ms\n\n", elapsed_h.count());

  // GPU
  cudaEvent_t start_d, stop_d;
  cudaEventCreate(&start_d);
  cudaEventCreate(&stop_d);

  cudaEventRecord(start_d);
  vecAdd_d(A, B, C_d, n);
  cudaEventRecord(stop_d);
  cudaEventSynchronize(stop_d);

  float elapsed_d = 0;
  cudaEventElapsedTime(&elapsed_d, start_d, stop_d);
  printf("GPU:\n----\n"); printVec(C_d, n);
  printf("vecAdd_d took %.4f ms\n", elapsed_d);

  cudaEventDestroy(start_d);
  cudaEventDestroy(stop_d);

  return 0;
}
