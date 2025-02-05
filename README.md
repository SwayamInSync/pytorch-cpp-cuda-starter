# pytorch-cpp-cuda-starter
Setting up VSCode to work with PyTorch in C/C++ with CUDA support

## Prerequisites
- CUDA Toolkit (tested with CUDA 12.4)
- CMake (>= 3.18)
- PyTorch with CUDA support
- VSCode with C/C++ and CMake Tools extensions
- GCC/G++ with C++17 support

## Installation Steps

### 1. Install CUDA Toolkit
Download and install CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)

### 2. Install CMake
```bash
# Remove old version (optional)
sudo apt remove cmake

# Install dependencies
sudo apt update && sudo apt install -y wget

# Download and install latest CMake
wget https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.sh
chmod +x cmake-3.28.3-linux-x86_64.sh
sudo mkdir /opt/cmake
sudo ./cmake-3.28.3-linux-x86_64.sh --skip-license --prefix=/opt/cmake
sudo ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

# Verify installation
cmake --version
```

### 3. Installing PyTorch
```bash
conda create -n cuda python=3.12
conda activate cuda
pip3 install torch torchvision torchaudio
conda install cudnn nccl
```

### 4. Setting up
- Run the `torch_info.py` to get the list of all necessary paths
- Replace the `<TORCH_PREFIX>` in `CMakeLists.txt` with the corresponding path from the above output
- Add the include paths from the `torch_info.py` output and put them inside `.vscode/c_cpp_properties.json`
- Reload the window for intellisense to work

### 5. Building the CUDA program
```bash
mkdir build
cd build
cmake ..
make
./my_cuda_program
```
