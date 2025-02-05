#include <iostream>
#include <cmath>
#include <torch/torch.h>
#include <cuda_runtime.h>

__global__ void kernel(float *input1, float *input2, float *output, int m, int n, int k)
{

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n)
  {
    float sum = 0;
    for(int i=0; i<k; i++)
    {
      sum += input1[row * k + i] * input2[i * n + col];
    }

    output[row*n+col] = sum;
  }
  
}

void kernelLauncher(float *input1, float * input2, float *output, int m, int n, int k)
{
  int threads = 16;
  dim3 dim_block(threads, threads);
  dim3 dim_grid(ceil(m / threads), ceil(n / threads));

  float *d_input1, *d_input2, *d_output;
  cudaMalloc((void **)&d_input1, m * k * sizeof(float));
  cudaMemcpy(d_input1, input1, m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_input2, k * n * sizeof(float));
  cudaMemcpy(d_input2, input2, k * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_output, m * n * sizeof(float));

  kernel<<<dim_grid, dim_block>>>(d_input1, d_input2, d_output, m, n, k);
  cudaMemcpy(output, d_output, m * n * sizeof(float), cudaMemcpyDeviceToHost);
}

int main()
{
  torch::manual_seed(0);
  int m = 64, n = 64, k = 64;
  auto input1 = torch::randn({m, k});
  auto input2 = torch::randn({k, n});
  auto output = torch::zeros({m, n});
  auto truth = torch::mm(input1, input2);
  kernelLauncher(input1.data_ptr<float>(), input2.data_ptr<float>(), output.data_ptr<float>(), m, n, k);
  std::cout << "Error: " << torch::sum(torch::abs(output - truth)) << std::endl;
}
