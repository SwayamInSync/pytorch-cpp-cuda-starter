cmake_minimum_required(VERSION 3.18)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 80)  # For A100

project(MyTorchProject CUDA CXX)

# Set C++17 standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA paths and directories
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17 -Xcompiler=-std=c++17")

# Set PyTorch paths
# Run "print(os.path.dirname(torch.__file__))" and put the output below
set(TORCH_INSTALL_PREFIX "Output of above command")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_INSTALL_PREFIX})
set(TORCH_INSTALL_DIR ${TORCH_INSTALL_PREFIX})

# Find PyTorch
find_package(Torch REQUIRED HINTS ${TORCH_INSTALL_DIR})

# Create executable
add_executable(my_cuda_program main.cu)

# Link libraries
target_link_libraries(my_cuda_program PRIVATE ${TORCH_LIBRARIES})

# Set properties
set_target_properties(my_cuda_program PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

set(CMAKE_BUILD_TYPE Release)
