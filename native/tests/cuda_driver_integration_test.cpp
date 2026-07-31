#include "flight4s/cuda/cuda_driver.hpp"

#include <cuda.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_success(
    const flight4s::cuda::CudaDriverStatus& status,
    const char* operation) {
  if (status.succeeded()) {
    return;
  }
  throw std::runtime_error(
      std::string(operation) + " failed with " +
      status.result_name + ": " + status.result_description +
      "\n" + status.error_log);
}

std::vector<std::uint8_t> read_file(const char* path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not open PTX test file");
  }
  return {
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>(),
  };
}

}  // namespace

int main(int argument_count, char** arguments) {
  if (argument_count != 2) {
    std::cerr << "expected PTX path\n";
    return 1;
  }

  const flight4s::cuda::CudaDriver driver;
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUstream stream = nullptr;
  CUdeviceptr memory = 0;
  bool retained = false;

  try {
    const auto context_result = driver.retain_primary_context(0);
    if (context_result.status.result_code == CUDA_ERROR_NO_DEVICE) {
      std::cout << "CUDA device unavailable; skipping\n";
      return 77;
    }
    require_success(
        context_result.status,
        "retain primary context");
    retained = true;
    context = context_result.context;

    if (context == nullptr) {
      throw std::runtime_error(
          "successful context retain returned null");
    }
    if (context_result.compute_capability_major <= 0) {
      throw std::runtime_error(
          "invalid compute capability metadata");
    }

    const auto stream_result = driver.create_stream(
        context,
        CU_STREAM_NON_BLOCKING);
    require_success(stream_result.status, "create stream");
    stream = stream_result.stream;
    if (stream == nullptr) {
      throw std::runtime_error(
          "successful stream creation returned null");
    }
    require_success(
        driver.synchronize_stream(context, stream),
        "synchronize stream");

    const std::array<std::int32_t, 4> host_source{
        11, 22, 33, 44};
    std::array<std::int32_t, 4> host_destination{};
    const auto memory_result = driver.allocate_device_memory(
        context,
        sizeof(host_source));
    require_success(
        memory_result.status,
        "allocate device memory");
    memory = memory_result.address;
    if (memory == 0) {
      throw std::runtime_error(
          "successful device allocation returned null");
    }

    require_success(
        driver.copy_host_to_device(
            context,
            memory,
            host_source.data(),
            sizeof(host_source)),
        "copy host to device");
    require_success(
        driver.copy_device_to_host(
            context,
            host_destination.data(),
            memory,
            sizeof(host_destination)),
        "copy device to host");
    if (host_destination != host_source) {
      throw std::runtime_error(
          "device-memory copy round trip did not preserve bytes");
    }

    const auto module_result =
        driver.load_ptx(context, read_file(arguments[1]));
    require_success(module_result.status, "load PTX module");
    module = module_result.module;
    if (module == nullptr) {
      throw std::runtime_error(
          "successful module load returned null");
    }

    const auto function_result = driver.resolve_function(
        context,
        module,
        "write_values");
    require_success(function_result.status, "resolve write_values");
    if (function_result.function == nullptr) {
      throw std::runtime_error(
          "successful function lookup returned null");
    }

    const auto missing_result = driver.resolve_function(
        context,
        module,
        "missing_kernel");
    if (missing_result.status.result_code != CUDA_ERROR_NOT_FOUND) {
      throw std::runtime_error(
          "missing function did not return CUDA_ERROR_NOT_FOUND");
    }

    const auto invalid_ptx = std::vector<std::uint8_t>{
        'n', 'o', 't', ' ', 'p', 't', 'x'};
    const auto invalid_result =
        driver.load_ptx(context, invalid_ptx);
    if (invalid_result.status.succeeded()) {
      throw std::runtime_error("invalid PTX unexpectedly loaded");
    }

    require_success(
        driver.unload_module(context, module),
        "unload module");
    module = nullptr;
    require_success(
        driver.free_device_memory(context, memory),
        "free device memory");
    memory = 0;
    require_success(
        driver.destroy_stream(context, stream),
        "destroy stream");
    stream = nullptr;
    require_success(
        driver.release_primary_context(0),
        "release primary context");
    retained = false;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    if (memory != 0 && context != nullptr) {
      static_cast<void>(
          driver.free_device_memory(context, memory));
    }
    if (module != nullptr && context != nullptr) {
      static_cast<void>(driver.unload_module(context, module));
    }
    if (stream != nullptr && context != nullptr) {
      static_cast<void>(driver.destroy_stream(context, stream));
    }
    if (retained) {
      static_cast<void>(driver.release_primary_context(0));
    }
    return 1;
  }

  std::cout << "CUDA Driver resource integration tests passed\n";
  return 0;
}
