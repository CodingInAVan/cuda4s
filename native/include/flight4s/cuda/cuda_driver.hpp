#pragma once

#include <cuda.h>

#include <cstdint>
#include <string>
#include <vector>

namespace flight4s::cuda {

struct CudaDriverStatus {
  CUresult result_code;
  std::string result_name;
  std::string result_description;
  std::string info_log;
  std::string error_log;

  [[nodiscard]] bool succeeded() const noexcept {
    return result_code == CUDA_SUCCESS;
  }
};

struct CudaContextResult {
  CudaDriverStatus status;
  CUcontext context;
  std::int32_t device_ordinal;
  std::int32_t compute_capability_major;
  std::int32_t compute_capability_minor;
};

struct CudaModuleResult {
  CudaDriverStatus status;
  CUmodule module;
};

struct CudaFunctionResult {
  CudaDriverStatus status;
  CUfunction function;
};

class CudaDriver final {
 public:
  [[nodiscard]] CudaContextResult retain_primary_context(
      std::int32_t device_ordinal) const;

  [[nodiscard]] CudaDriverStatus release_primary_context(
      std::int32_t device_ordinal) const;

  [[nodiscard]] CudaModuleResult load_ptx(
      CUcontext context,
      const std::vector<std::uint8_t>& ptx) const;

  [[nodiscard]] CudaDriverStatus unload_module(
      CUcontext context,
      CUmodule module) const;

  [[nodiscard]] CudaFunctionResult resolve_function(
      CUcontext context,
      CUmodule module,
      const std::string& name) const;
};

}  // namespace flight4s::cuda
