cmake_minimum_required(VERSION 3.18)
project(CudaJitJNI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find JNI
find_package(JNI REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Define the shared library
add_library(cudajitjni SHARED CudaJitJNI.cpp)

# Include directories
target_include_directories(cudajitjni PRIVATE
    ${JNI_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(cudajitjni PRIVATE
    CUDA::cuda_driver
    CUDA::nvrtc
)

# Output name should be libcudajitjni.so
set_target_properties(cudajitjni PROPERTIES PREFIX "lib")
