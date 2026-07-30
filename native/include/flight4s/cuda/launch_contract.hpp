#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace flight4s::cuda {

struct LaunchGeometry {
  std::int32_t grid_x;
  std::int32_t grid_y;
  std::int32_t grid_z;
  std::int32_t block_x;
  std::int32_t block_y;
  std::int32_t block_z;
  std::int32_t dynamic_shared_memory_bytes;
  bool uses_cluster;
  std::int32_t cluster_x;
  std::int32_t cluster_y;
  std::int32_t cluster_z;
};

struct ArgumentLayout {
  void* storage;
  std::int64_t storage_size_bytes;
  const std::int32_t* offsets;
  std::size_t offset_count;
  const std::int8_t* descriptor_codes;
  std::size_t descriptor_count;
};

std::optional<std::string> validate_launch_geometry(
    const LaunchGeometry& geometry);

std::optional<std::string> validate_argument_layout(
    const ArgumentLayout& layout);

std::vector<void*> build_argument_pointers(const ArgumentLayout& layout);

}  // namespace flight4s::cuda
