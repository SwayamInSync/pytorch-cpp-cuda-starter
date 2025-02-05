import torch
import torch.utils.cpp_extension
import os

def print_section(title, content):
    print(f"\n{'-'*20} {title} {'-'*20}")
    if isinstance(content, (list, tuple)):
        for item in content:
            print(item)
    else:
        print(content)

print_section("CMake Paths", [
    f"CMake Prefix Path: {torch.utils.cmake_prefix_path}",
    f"Torch CMake Dir: {os.path.join(os.path.dirname(torch.__file__), 'share', 'cmake')}"
])

print_section("PyTorch Paths", [
    f"Torch Installation Dir: {torch.__path__[0]}",
    f"Torch Include Dirs: {torch.utils.cpp_extension.include_paths()}"
])

# CUDA information if available
if torch.cuda.is_available():
    print_section("CUDA Information", [
        f"CUDA Available: {torch.cuda.is_available()}",
        f"CUDA Version: {torch.version.cuda}",
        f"Current CUDA Device: {torch.cuda.current_device()}",
        f"Device Name: {torch.cuda.get_device_name(0)}",
        f"Device Count: {torch.cuda.device_count()}"
    ])

print_section("PyTorch Build Information", [
    f"PyTorch Version: {torch.__version__}",
    f"Debug Build: {torch.version.debug}",
    f"C++ ABI: {'New' if torch._C._GLIBCXX_USE_CXX11_ABI else 'Old'}"
])

print_section("Library Paths", [
    f"Library Path: {os.path.join(os.path.dirname(torch.__file__), 'lib')}",
    f"Library Dependencies: {os.path.join(os.path.dirname(torch.__file__), 'lib', 'libtorch.so')}"
])
