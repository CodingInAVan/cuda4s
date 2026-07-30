#include "flight4s/cuda/cuda_launcher.hpp"

#include <stdexcept>

namespace flight4s::cuda {

CUresult launch_kernel(const CudaLaunchRequest& request) {
  if (request.function == nullptr) {
    throw std::invalid_argument("CUDA function handle must not be null");
  }
  if (auto error = validate_launch_geometry(request.geometry)) {
    throw std::invalid_argument(*error);
  }
  if (auto error = validate_argument_layout(request.arguments)) {
    throw std::invalid_argument(*error);
  }

  auto argument_pointers = build_argument_pointers(request.arguments);

  CUlaunchConfig config{};
  config.gridDimX = static_cast<unsigned int>(request.geometry.grid_x);
  config.gridDimY = static_cast<unsigned int>(request.geometry.grid_y);
  config.gridDimZ = static_cast<unsigned int>(request.geometry.grid_z);
  config.blockDimX = static_cast<unsigned int>(request.geometry.block_x);
  config.blockDimY = static_cast<unsigned int>(request.geometry.block_y);
  config.blockDimZ = static_cast<unsigned int>(request.geometry.block_z);
  config.sharedMemBytes = static_cast<unsigned int>(
      request.geometry.dynamic_shared_memory_bytes);
  config.hStream = request.stream;

  CUlaunchAttribute cluster_attribute{};
  if (request.geometry.uses_cluster) {
    cluster_attribute.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    cluster_attribute.value.clusterDim.x =
        static_cast<unsigned int>(request.geometry.cluster_x);
    cluster_attribute.value.clusterDim.y =
        static_cast<unsigned int>(request.geometry.cluster_y);
    cluster_attribute.value.clusterDim.z =
        static_cast<unsigned int>(request.geometry.cluster_z);
    config.attrs = &cluster_attribute;
    config.numAttrs = 1;
  }

  return cuLaunchKernelEx(
      &config,
      request.function,
      argument_pointers.empty() ? nullptr : argument_pointers.data(),
      nullptr);
}

}  // namespace flight4s::cuda
