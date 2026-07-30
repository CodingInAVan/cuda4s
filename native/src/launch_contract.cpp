#include "flight4s/cuda/launch_contract.hpp"

#include <cstdint>
#include <sstream>

namespace flight4s::cuda {
namespace {

struct DescriptorLayout {
  std::size_t size_bytes;
  std::size_t alignment_bytes;
};

std::optional<DescriptorLayout> descriptor_layout(std::int8_t code) {
  switch (static_cast<std::uint8_t>(code)) {
    case 1:
      return DescriptorLayout{1, 1};
    case 2:
    case 3:
    case 6:
      return DescriptorLayout{4, 4};
    case 4:
    case 5:
      return DescriptorLayout{2, 2};
    case 7:
    case 10:
      return DescriptorLayout{8, 8};
    case 8:
    case 9:
      return DescriptorLayout{1, 1};
    default:
      return std::nullopt;
  }
}

std::optional<std::string> require_positive(
    const std::string& name,
    std::int32_t value) {
  if (value > 0) {
    return std::nullopt;
  }
  std::ostringstream message;
  message << name << " must be positive: " << value;
  return message.str();
}

}  // namespace

std::optional<std::string> validate_launch_geometry(
    const LaunchGeometry& geometry) {
  const struct {
    const char* name;
    std::int32_t value;
  } dimensions[] = {
      {"grid x dimension", geometry.grid_x},
      {"grid y dimension", geometry.grid_y},
      {"grid z dimension", geometry.grid_z},
      {"block x dimension", geometry.block_x},
      {"block y dimension", geometry.block_y},
      {"block z dimension", geometry.block_z},
  };

  for (const auto& dimension : dimensions) {
    if (auto error = require_positive(dimension.name, dimension.value)) {
      return error;
    }
  }

  if (geometry.dynamic_shared_memory_bytes < 0) {
    std::ostringstream message;
    message << "dynamic shared memory must not be negative: "
            << geometry.dynamic_shared_memory_bytes;
    return message.str();
  }

  if (!geometry.uses_cluster) {
    if (geometry.cluster_x != 0 || geometry.cluster_y != 0 ||
        geometry.cluster_z != 0) {
      return "cluster dimensions must be zero when clusters are disabled";
    }
    return std::nullopt;
  }

  const struct {
    const char* name;
    std::int32_t grid;
    std::int32_t cluster;
  } cluster_dimensions[] = {
      {"x", geometry.grid_x, geometry.cluster_x},
      {"y", geometry.grid_y, geometry.cluster_y},
      {"z", geometry.grid_z, geometry.cluster_z},
  };

  for (const auto& dimension : cluster_dimensions) {
    if (auto error = require_positive(
            std::string("cluster ") + dimension.name + " dimension",
            dimension.cluster)) {
      return error;
    }
    if (dimension.grid % dimension.cluster != 0) {
      std::ostringstream message;
      message << "grid " << dimension.name << " dimension "
              << dimension.grid << " must be divisible by cluster "
              << dimension.name << " dimension " << dimension.cluster;
      return message.str();
    }
  }

  return std::nullopt;
}

std::optional<std::string> validate_argument_layout(
    const ArgumentLayout& layout) {
  if (layout.storage_size_bytes < 0) {
    return "argument storage must be a direct buffer";
  }
  if (layout.storage_size_bytes > 0 && layout.storage == nullptr) {
    return "argument storage address must not be null";
  }
  if (layout.offset_count != layout.descriptor_count) {
    std::ostringstream message;
    message << "argument metadata count mismatch: " << layout.offset_count
            << " offsets and " << layout.descriptor_count << " descriptors";
    return message.str();
  }
  if (layout.offset_count > 0 &&
      (layout.offsets == nullptr || layout.descriptor_codes == nullptr)) {
    return "argument metadata arrays must not be null";
  }

  std::size_t max_alignment = 1;
  std::vector<DescriptorLayout> descriptors;
  descriptors.reserve(layout.descriptor_count);
  for (std::size_t index = 0; index < layout.descriptor_count; ++index) {
    auto descriptor = descriptor_layout(layout.descriptor_codes[index]);
    if (!descriptor) {
      std::ostringstream message;
      message << "unknown descriptor code at argument " << index << ": "
              << static_cast<int>(
                     static_cast<std::uint8_t>(
                         layout.descriptor_codes[index]));
      return message.str();
    }
    if (descriptor->alignment_bytes > max_alignment) {
      max_alignment = descriptor->alignment_bytes;
    }
    descriptors.push_back(*descriptor);
  }

  if (layout.storage_size_bytes > 0 &&
      reinterpret_cast<std::uintptr_t>(layout.storage) % max_alignment != 0) {
    std::ostringstream message;
    message << "argument storage base is not aligned to " << max_alignment
            << " bytes";
    return message.str();
  }

  std::int64_t previous_end = 0;
  for (std::size_t index = 0; index < layout.offset_count; ++index) {
    const auto offset = static_cast<std::int64_t>(layout.offsets[index]);
    const auto descriptor = descriptors[index];
    if (offset < 0) {
      std::ostringstream message;
      message << "argument " << index << " has negative offset " << offset;
      return message.str();
    }
    if (offset % static_cast<std::int64_t>(descriptor.alignment_bytes) != 0) {
      std::ostringstream message;
      message << "argument " << index << " offset " << offset
              << " is not aligned to " << descriptor.alignment_bytes
              << " bytes";
      return message.str();
    }
    if (offset < previous_end) {
      std::ostringstream message;
      message << "argument " << index << " overlaps the previous argument";
      return message.str();
    }

    const auto end =
        offset + static_cast<std::int64_t>(descriptor.size_bytes);
    if (end > layout.storage_size_bytes) {
      std::ostringstream message;
      message << "argument " << index << " ends at " << end
              << " beyond storage size " << layout.storage_size_bytes;
      return message.str();
    }
    previous_end = end;
  }

  if (previous_end != layout.storage_size_bytes) {
    std::ostringstream message;
    message << "argument layout ends at " << previous_end
            << " but storage size is " << layout.storage_size_bytes;
    return message.str();
  }

  return std::nullopt;
}

std::vector<void*> build_argument_pointers(const ArgumentLayout& layout) {
  auto* base = static_cast<std::byte*>(layout.storage);
  std::vector<void*> pointers;
  pointers.reserve(layout.offset_count);
  for (std::size_t index = 0; index < layout.offset_count; ++index) {
    pointers.push_back(base + layout.offsets[index]);
  }
  return pointers;
}

}  // namespace flight4s::cuda
