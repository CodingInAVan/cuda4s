#include "flight4s/cuda/cuda_driver.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flight4s::cuda {
namespace {

constexpr std::size_t jit_log_capacity = 64 * 1024;

CudaDriverStatus status_for(
    CUresult result,
    std::string info_log = {},
    std::string error_log = {}) {
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

void require_handle(const void* handle, const char* description) {
  if (handle == nullptr) {
    throw std::invalid_argument(
        std::string(description) + " must not be null");
  }
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

class CurrentContextScope final {
 public:
  explicit CurrentContextScope(CUcontext context)
      : context_(context),
        push_result_(cuCtxPushCurrent(context)),
        active_(push_result_ == CUDA_SUCCESS) {}

  CurrentContextScope(const CurrentContextScope&) = delete;
  CurrentContextScope& operator=(const CurrentContextScope&) = delete;
  CurrentContextScope(CurrentContextScope&&) = delete;
  CurrentContextScope& operator=(CurrentContextScope&&) = delete;

  ~CurrentContextScope() {
    if (active_) {
      CUcontext popped = nullptr;
      cuCtxPopCurrent(&popped);
    }
  }

  [[nodiscard]] CUresult push_result() const noexcept {
    return push_result_;
  }

  [[nodiscard]] CUresult close() noexcept {
    if (!active_) {
      return push_result_;
    }

    CUcontext popped = nullptr;
    const auto result = cuCtxPopCurrent(&popped);
    active_ = false;
    if (result != CUDA_SUCCESS) {
      return result;
    }
    return popped == context_ ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CONTEXT;
  }

 private:
  CUcontext context_;
  CUresult push_result_;
  bool active_;
};

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
  return status_for(
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
    return {status_for(result), nullptr, device_ordinal, 0, 0};
  }

  CUdevice device = 0;
  result = cuDeviceGet(&device, device_ordinal);
  if (result != CUDA_SUCCESS) {
    return {status_for(result), nullptr, device_ordinal, 0, 0};
  }

  int major = 0;
  result = cuDeviceGetAttribute(
      &major,
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
      device);
  if (result != CUDA_SUCCESS) {
    return {status_for(result), nullptr, device_ordinal, 0, 0};
  }

  int minor = 0;
  result = cuDeviceGetAttribute(
      &minor,
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
      device);
  if (result != CUDA_SUCCESS) {
    return {status_for(result), nullptr, device_ordinal, major, 0};
  }

  CUcontext context = nullptr;
  result = cuDevicePrimaryCtxRetain(&context, device);
  return {
      status_for(result),
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
    return status_for(device_result);
  }
  return status_for(cuDevicePrimaryCtxRelease(device));
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
    return {status_for(current.push_result()), nullptr};
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
    return status_for(current.push_result());
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
    return {status_for(current.push_result()), nullptr};
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

}  // namespace flight4s::cuda
