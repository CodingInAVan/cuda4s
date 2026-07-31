#include "flight4s/cuda/cuda_launcher.hpp"
#include "flight4s/cuda/current_context_scope.hpp"

#include <stdexcept>

namespace flight4s::cuda {

CudaDriverStatus CudaLauncher::launch(
    const CudaLaunchRequest& request) const {
  if (request.context == nullptr) {
    throw std::invalid_argument("CUDA context handle must not be null");
  }
  if (request.function == nullptr) {
    throw std::invalid_argument("CUDA function handle must not be null");
  }
  if (auto error = validate_launch_geometry(request.geometry)) {
    throw std::invalid_argument(*error);
  }
  if (auto error = validate_argument_layout(request.arguments)) {
    throw std::invalid_argument(*error);
  }

  CurrentContextScope current(request.context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
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

  const auto launch_result = cuLaunchKernelEx(
      &config,
      request.function,
      argument_pointers.empty() ? nullptr : argument_pointers.data(),
      nullptr);
  const auto pop_result = current.close();
  return make_driver_status(
      launch_result == CUDA_SUCCESS ? pop_result : launch_result);
}

}  // namespace flight4s::cuda
