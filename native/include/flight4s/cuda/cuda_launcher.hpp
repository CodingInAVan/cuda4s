#pragma once

#include <cuda.h>

#include "flight4s/cuda/cuda_driver.hpp"
#include "flight4s/cuda/launch_contract.hpp"

namespace flight4s::cuda {

struct CudaLaunchRequest {
  CUcontext context;
  CUfunction function;
  CUstream stream;
  LaunchGeometry geometry;
  ArgumentLayout arguments;
};

class CudaLauncher final {
 public:
  [[nodiscard]] CudaDriverStatus launch(
      const CudaLaunchRequest& request) const;
};

}  // namespace flight4s::cuda
