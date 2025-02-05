#include <iostream>
#include <torch/torch.h>
#include <cuda_runtime.h>

int main() {
    try {
        int num_devices;
        cudaGetDeviceCount(&num_devices);
        if (num_devices > 0) {
            std::cout << "CUDA Devices found: " << num_devices << std::endl;
        } else {
            std::cerr << "No CUDA devices found!" << std::endl;
            return 1;
        }

        torch::Device device(torch::kCUDA, 0); // Use CUDA device 0

        auto tensor = torch::randn({2, 3}, device);

        std::cout << tensor << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
