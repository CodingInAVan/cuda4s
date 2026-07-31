#include "flight4s/cuda/cuda_driver.hpp"
#include "flight4s/cuda/current_context_scope.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flight4s::cuda {

CudaDriverStatus make_driver_status(
    CUresult result,
    std::string info_log,
    std::string error_log) {
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  return {
      result,
      name == nullptr ? "CUDA_ERROR_UNKNOWN" : name,
      description == nullptr ? "no CUDA error description" : description,
      std::move(info_log),
      std::move(error_log),
  };
}

namespace {

constexpr std::size_t jit_log_capacity = 64 * 1024;

void require_handle(const void* handle, const char* description) {
  if (handle == nullptr) {
    throw std::invalid_argument(
        std::string(description) + " must not be null");
  }
}

void require_device_address(
    CUdeviceptr address,
    const char* description) {
  if (address == 0) {
    throw std::invalid_argument(
        std::string(description) + " must not be null");
  }
}

void require_stream_flags(std::uint32_t flags) {
  if (flags != CU_STREAM_DEFAULT &&
      flags != CU_STREAM_NON_BLOCKING) {
    throw std::invalid_argument(
        "unsupported CUDA stream flags");
  }
}

std::size_t checked_size(
    std::uint64_t size_bytes,
    const char* description) {
  if (size_bytes == 0) {
    throw std::invalid_argument(
        std::string(description) + " must be positive");
  }
  if (size_bytes > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error(
        std::string(description) + " exceeds the native size limit");
  }
  return static_cast<std::size_t>(size_bytes);
}

void require_c_string(
    std::string_view value,
    const char* description) {
  if (value.empty()) {
    throw std::invalid_argument(
        std::string(description) + " must not be empty");
  }
  if (value.find('\0') != std::string_view::npos) {
    throw std::invalid_argument(
        std::string(description) + " must not contain a null byte");
  }
}

std::string log_text(const std::vector<char>& buffer) {
  const auto end = std::find(buffer.begin(), buffer.end(), '\0');
  return std::string(buffer.begin(), end);
}

CudaDriverStatus finish_context_operation(
    CUresult operation_result,
    CurrentContextScope& scope,
    std::string info_log = {},
    std::string error_log = {}) {
  const auto pop_result = scope.close();
  const auto final_result =
      operation_result == CUDA_SUCCESS ? pop_result : operation_result;
  return make_driver_status(
      final_result,
      std::move(info_log),
      std::move(error_log));
}

}  // namespace

CudaContextResult CudaDriver::retain_primary_context(
    std::int32_t device_ordinal) const {
  if (device_ordinal < 0) {
    throw std::invalid_argument(
        "CUDA device ordinal must not be negative");
  }

  auto result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    return {make_driver_status(result), nullptr, device_ordinal, 0, 0};
  }

  CUdevice device = 0;
  result = cuDeviceGet(&device, device_ordinal);
  if (result != CUDA_SUCCESS) {
    return {make_driver_status(result), nullptr, device_ordinal, 0, 0};
  }

  int major = 0;
  result = cuDeviceGetAttribute(
      &major,
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
      device);
  if (result != CUDA_SUCCESS) {
    return {make_driver_status(result), nullptr, device_ordinal, 0, 0};
  }

  int minor = 0;
  result = cuDeviceGetAttribute(
      &minor,
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
      device);
  if (result != CUDA_SUCCESS) {
    return {make_driver_status(result), nullptr, device_ordinal, major, 0};
  }

  CUcontext context = nullptr;
  result = cuDevicePrimaryCtxRetain(&context, device);
  return {
      make_driver_status(result),
      result == CUDA_SUCCESS ? context : nullptr,
      device_ordinal,
      major,
      minor,
  };
}

CudaDriverStatus CudaDriver::release_primary_context(
    std::int32_t device_ordinal) const {
  if (device_ordinal < 0) {
    throw std::invalid_argument(
        "CUDA device ordinal must not be negative");
  }

  CUdevice device = 0;
  const auto device_result = cuDeviceGet(&device, device_ordinal);
  if (device_result != CUDA_SUCCESS) {
    return make_driver_status(device_result);
  }
  return make_driver_status(cuDevicePrimaryCtxRelease(device));
}

CudaModuleResult CudaDriver::load_ptx(
    CUcontext context,
    const std::vector<std::uint8_t>& ptx) const {
  require_handle(context, "CUDA context handle");
  if (ptx.empty()) {
    throw std::invalid_argument("PTX must not be empty");
  }
  if (std::find(ptx.begin(), ptx.end(), std::uint8_t{0}) != ptx.end()) {
    throw std::invalid_argument("PTX must not contain a null byte");
  }

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return {make_driver_status(current.push_result()), nullptr};
  }

  std::vector<char> null_terminated_ptx(ptx.begin(), ptx.end());
  null_terminated_ptx.push_back('\0');
  std::vector<char> info_log(jit_log_capacity, '\0');
  std::vector<char> error_log(jit_log_capacity, '\0');
  std::array<CUjit_option, 4> options{
      CU_JIT_INFO_LOG_BUFFER,
      CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
      CU_JIT_ERROR_LOG_BUFFER,
      CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
  };
  std::array<void*, 4> option_values{
      info_log.data(),
      reinterpret_cast<void*>(
          static_cast<std::uintptr_t>(info_log.size())),
      error_log.data(),
      reinterpret_cast<void*>(
          static_cast<std::uintptr_t>(error_log.size())),
  };

  CUmodule module = nullptr;
  const auto load_result = cuModuleLoadDataEx(
      &module,
      null_terminated_ptx.data(),
      static_cast<unsigned int>(options.size()),
      options.data(),
      option_values.data());
  auto status = finish_context_operation(
      load_result,
      current,
      log_text(info_log),
      log_text(error_log));
  const bool succeeded = status.succeeded();
  return {
      std::move(status),
      succeeded ? module : nullptr,
  };
}

CudaDriverStatus CudaDriver::unload_module(
    CUcontext context,
    CUmodule module) const {
  require_handle(context, "CUDA context handle");
  require_handle(module, "CUDA module handle");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(cuModuleUnload(module), current);
}

CudaFunctionResult CudaDriver::resolve_function(
    CUcontext context,
    CUmodule module,
    const std::string& name) const {
  require_handle(context, "CUDA context handle");
  require_handle(module, "CUDA module handle");
  require_c_string(name, "CUDA function name");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return {make_driver_status(current.push_result()), nullptr};
  }

  CUfunction function = nullptr;
  const auto resolve_result =
      cuModuleGetFunction(&function, module, name.c_str());
  auto status =
      finish_context_operation(resolve_result, current);
  const bool succeeded = status.succeeded();
  return {
      std::move(status),
      succeeded ? function : nullptr,
  };
}

CudaStreamResult CudaDriver::create_stream(
    CUcontext context,
    std::uint32_t flags) const {
  require_handle(context, "CUDA context handle");
  require_stream_flags(flags);

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return {make_driver_status(current.push_result()), nullptr};
  }

  CUstream stream = nullptr;
  const auto create_result = cuStreamCreate(&stream, flags);
  auto status = finish_context_operation(create_result, current);
  const bool succeeded = status.succeeded();
  return {
      std::move(status),
      succeeded ? stream : nullptr,
  };
}

CudaDriverStatus CudaDriver::destroy_stream(
    CUcontext context,
    CUstream stream) const {
  require_handle(context, "CUDA context handle");
  require_handle(stream, "CUDA stream handle");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(cuStreamDestroy(stream), current);
}

CudaDriverStatus CudaDriver::synchronize_stream(
    CUcontext context,
    CUstream stream) const {
  require_handle(context, "CUDA context handle");
  require_handle(stream, "CUDA stream handle");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(
      cuStreamSynchronize(stream),
      current);
}

CudaPinnedMemoryResult CudaDriver::allocate_pinned_memory(
    CUcontext context,
    std::uint64_t size_bytes) const {
  require_handle(context, "CUDA context handle");
  const auto native_size =
      checked_size(size_bytes, "CUDA pinned allocation size");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return {make_driver_status(current.push_result()), nullptr};
  }

  void* address = nullptr;
  const auto allocation_result = cuMemHostAlloc(
      &address,
      native_size,
      0);
  auto status = finish_context_operation(allocation_result, current);
  const bool succeeded = status.succeeded();
  return {
      std::move(status),
      succeeded ? address : nullptr,
  };
}

CudaDriverStatus CudaDriver::free_pinned_memory(
    CUcontext context,
    void* address) const {
  require_handle(context, "CUDA context handle");
  require_handle(address, "CUDA pinned host address");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(cuMemFreeHost(address), current);
}

CudaDeviceMemoryResult CudaDriver::allocate_device_memory(
    CUcontext context,
    std::uint64_t size_bytes) const {
  require_handle(context, "CUDA context handle");
  const auto native_size =
      checked_size(size_bytes, "CUDA allocation size");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return {make_driver_status(current.push_result()), 0};
  }

  CUdeviceptr address = 0;
  const auto allocation_result = cuMemAlloc(&address, native_size);
  auto status =
      finish_context_operation(allocation_result, current);
  const bool succeeded = status.succeeded();
  return {
      std::move(status),
      succeeded ? address : 0,
  };
}

CudaDriverStatus CudaDriver::free_device_memory(
    CUcontext context,
    CUdeviceptr address) const {
  require_handle(context, "CUDA context handle");
  require_device_address(address, "CUDA device address");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(cuMemFree(address), current);
}

CudaDriverStatus CudaDriver::copy_host_to_device(
    CUcontext context,
    CUdeviceptr destination,
    const void* source,
    std::uint64_t size_bytes) const {
  require_handle(context, "CUDA context handle");
  require_device_address(destination, "CUDA destination address");
  require_handle(source, "host source address");
  const auto native_size =
      checked_size(size_bytes, "CUDA copy size");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(
      cuMemcpyHtoD(destination, source, native_size),
      current);
}

CudaDriverStatus CudaDriver::copy_device_to_host(
    CUcontext context,
    void* destination,
    CUdeviceptr source,
    std::uint64_t size_bytes) const {
  require_handle(context, "CUDA context handle");
  require_handle(destination, "host destination address");
  require_device_address(source, "CUDA source address");
  const auto native_size =
      checked_size(size_bytes, "CUDA copy size");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(
      cuMemcpyDtoH(destination, source, native_size),
      current);
}

}  // namespace flight4s::cuda
