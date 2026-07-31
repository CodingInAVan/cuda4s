#include "flight4s/cuda/cuda_launcher.hpp"

#include <cuda.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  throw std::runtime_error(
      std::string(operation) + " failed with " +
      (name == nullptr ? "unknown CUDA error" : name) + ": " +
      (description == nullptr ? "no description" : description));
}

void write_argument(
    std::byte* destination,
    const void* source,
    std::size_t size) {
  std::memcpy(destination, source, size);
}

void launch_and_verify(
    CUcontext context,
    CUfunction function,
    CUdeviceptr output,
    bool uses_cluster,
    int base_value) {
  static_assert(
      sizeof(CUdeviceptr) == sizeof(std::uint64_t),
      "Flight4s currently requires 64-bit CUDA device pointers");

  alignas(8) std::array<std::byte, 12> storage{};
  write_argument(storage.data(), &output, sizeof(output));
  write_argument(storage.data() + 8, &base_value, sizeof(base_value));

  const std::array<std::int32_t, 2> offsets{0, 8};
  const std::array<std::int8_t, 2> descriptor_codes{10, 2};
  const flight4s::cuda::LaunchGeometry geometry{
      2, 1, 1,
      32, 1, 1,
      0,
      uses_cluster,
      uses_cluster ? 2 : 0,
      uses_cluster ? 1 : 0,
      uses_cluster ? 1 : 0,
  };
  const flight4s::cuda::ArgumentLayout arguments{
      storage.data(),
      static_cast<std::int64_t>(storage.size()),
      offsets.data(),
      offsets.size(),
      descriptor_codes.data(),
      descriptor_codes.size(),
  };

  const flight4s::cuda::CudaLauncher launcher;
  check_cuda(cuCtxSetCurrent(nullptr), "clear current context");
  const auto launch_status = launcher.launch(
      {context, function, nullptr, geometry, arguments});
  check_cuda(
      launch_status.result_code,
      uses_cluster ? "clustered cuLaunchKernelEx"
                   : "ordinary cuLaunchKernelEx");
  CUcontext current = nullptr;
  check_cuda(cuCtxGetCurrent(&current), "query current context");
  if (current != nullptr) {
    throw std::runtime_error(
        "CUDA launcher did not restore the previous context");
  }
  check_cuda(cuCtxSetCurrent(context), "restore current context");

  std::vector<int> output_values(64);
  check_cuda(
      cuMemcpyDtoH(
          output_values.data(),
          output,
          output_values.size() * sizeof(int)),
      "cuMemcpyDtoH");
  for (std::size_t index = 0; index < output_values.size(); ++index) {
    const int expected = base_value + static_cast<int>(index);
    if (output_values[index] != expected) {
      throw std::runtime_error(
          "kernel output mismatch at index " + std::to_string(index));
    }
  }
}

}  // namespace

int main(int argument_count, char** arguments) {
  if (argument_count != 2) {
    std::cerr << "expected PTX path\n";
    return 1;
  }

  CUdevice device = 0;
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUdeviceptr output = 0;

  try {
    const auto init_result = cuInit(0);
    if (init_result == CUDA_ERROR_NO_DEVICE) {
      std::cout << "CUDA device unavailable; skipping\n";
      return 77;
    }
    check_cuda(init_result, "cuInit");
    check_cuda(cuDeviceGet(&device, 0), "cuDeviceGet");

    int compute_capability_major = 0;
    check_cuda(
        cuDeviceGetAttribute(
            &compute_capability_major,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device),
        "cuDeviceGetAttribute");
    if (compute_capability_major < 9) {
      std::cout << "cluster launch requires compute capability 9.0; "
                << "skipping\n";
      return 77;
    }

    check_cuda(
        cuDevicePrimaryCtxRetain(&context, device),
        "cuDevicePrimaryCtxRetain");
    check_cuda(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    check_cuda(cuModuleLoad(&module, arguments[1]), "cuModuleLoad");

    CUfunction function = nullptr;
    check_cuda(
        cuModuleGetFunction(&function, module, "write_values"),
        "cuModuleGetFunction");
    check_cuda(
        cuMemAlloc(&output, 64 * sizeof(int)),
        "cuMemAlloc");

    launch_and_verify(context, function, output, false, 100);
    launch_and_verify(context, function, output, true, 200);

    check_cuda(cuMemFree(output), "cuMemFree");
    output = 0;
    check_cuda(cuModuleUnload(module), "cuModuleUnload");
    module = nullptr;
    check_cuda(
        cuDevicePrimaryCtxRelease(device),
        "cuDevicePrimaryCtxRelease");
    context = nullptr;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    if (output != 0) {
      cuMemFree(output);
    }
    if (module != nullptr) {
      cuModuleUnload(module);
    }
    if (context != nullptr) {
      cuDevicePrimaryCtxRelease(device);
    }
    return 1;
  }

  std::cout << "CUDA launch integration tests passed\n";
  return 0;
}
