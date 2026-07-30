#include "flight4s/cuda/launch_contract.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    ++failures;
  }
}

void expect_error_contains(
    const std::optional<std::string>& error,
    const std::string& expected) {
  expect(error.has_value(), "expected validation failure");
  if (error) {
    expect(
        error->find(expected) != std::string::npos,
        "validation failure did not contain expected text");
  }
}

}  // namespace

int main() {
  using flight4s::cuda::ArgumentLayout;
  using flight4s::cuda::LaunchGeometry;

  LaunchGeometry ordinary{
      4, 2, 1,
      32, 8, 1,
      2048,
      false,
      0, 0, 0,
  };
  expect(
      !flight4s::cuda::validate_launch_geometry(ordinary),
      "ordinary launch geometry should be valid");

  LaunchGeometry clustered{
      4, 2, 1,
      32, 8, 1,
      0,
      true,
      2, 2, 1,
  };
  expect(
      !flight4s::cuda::validate_launch_geometry(clustered),
      "clustered launch geometry should be valid");

  auto invalid_cluster = clustered;
  invalid_cluster.cluster_x = 3;
  expect_error_contains(
      flight4s::cuda::validate_launch_geometry(invalid_cluster),
      "must be divisible");

  auto unexpected_cluster = ordinary;
  unexpected_cluster.cluster_x = 1;
  expect_error_contains(
      flight4s::cuda::validate_launch_geometry(unexpected_cluster),
      "must be zero");

  alignas(8) std::array<std::byte, 24> storage{};
  const std::array<std::int32_t, 3> offsets{0, 8, 16};
  const std::array<std::int8_t, 3> descriptor_codes{2, 7, 10};
  ArgumentLayout valid_layout{
      storage.data(),
      static_cast<std::int64_t>(storage.size()),
      offsets.data(),
      offsets.size(),
      descriptor_codes.data(),
      descriptor_codes.size(),
  };

  expect(
      !flight4s::cuda::validate_argument_layout(valid_layout),
      "aligned argument layout should be valid");
  const auto pointers =
      flight4s::cuda::build_argument_pointers(valid_layout);
  expect(pointers.size() == 3, "argument pointer count should match");
  expect(
      pointers[0] == storage.data(),
      "first argument pointer should use the storage base");
  expect(
      pointers[1] == storage.data() + 8,
      "second argument pointer should use its offset");
  expect(
      pointers[2] == storage.data() + 16,
      "third argument pointer should use its offset");

  const struct {
    std::int8_t code;
    std::int64_t size;
  } descriptor_sizes[] = {
      {1, 1},
      {2, 4},
      {3, 4},
      {4, 2},
      {5, 2},
      {6, 4},
      {7, 8},
      {8, 1},
      {9, 1},
      {10, 8},
  };
  const std::int32_t zero_offset = 0;
  for (const auto& descriptor : descriptor_sizes) {
    const ArgumentLayout single_argument{
        storage.data(),
        descriptor.size,
        &zero_offset,
        1,
        &descriptor.code,
        1,
    };
    expect(
        !flight4s::cuda::validate_argument_layout(single_argument),
        "every stable descriptor code should have the expected size");
  }

  const std::array<std::int8_t, 3> unknown_codes{2, 127, 10};
  auto unknown_layout = valid_layout;
  unknown_layout.descriptor_codes = unknown_codes.data();
  expect_error_contains(
      flight4s::cuda::validate_argument_layout(unknown_layout),
      "unknown descriptor code");

  const std::array<std::int32_t, 3> overlapping_offsets{0, 4, 16};
  const std::array<std::int8_t, 3> overlapping_codes{7, 2, 10};
  auto overlapping_layout = valid_layout;
  overlapping_layout.offsets = overlapping_offsets.data();
  overlapping_layout.descriptor_codes = overlapping_codes.data();
  expect_error_contains(
      flight4s::cuda::validate_argument_layout(overlapping_layout),
      "overlaps");

  auto count_mismatch = valid_layout;
  count_mismatch.descriptor_count = 2;
  expect_error_contains(
      flight4s::cuda::validate_argument_layout(count_mismatch),
      "count mismatch");

  auto trailing_storage = valid_layout;
  trailing_storage.storage_size_bytes = 32;
  expect_error_contains(
      flight4s::cuda::validate_argument_layout(trailing_storage),
      "layout ends");

  if (failures != 0) {
    std::cerr << failures << " test assertion(s) failed\n";
    return 1;
  }
  std::cout << "launch contract tests passed\n";
  return 0;
}
