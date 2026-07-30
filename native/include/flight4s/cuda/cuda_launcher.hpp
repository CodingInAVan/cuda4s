#pragma once

#include <cuda.h>

#include "flight4s/cuda/launch_contract.hpp"

namespace flight4s::cuda {

struct CudaLaunchRequest {
  CUfunction function;
  CUstream stream;
  LaunchGeometry geometry;
  ArgumentLayout arguments;
};

CUresult launch_kernel(const CudaLaunchRequest& request);

}  // namespace flight4s::cuda
