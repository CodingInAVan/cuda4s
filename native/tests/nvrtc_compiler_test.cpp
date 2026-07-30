#include "flight4s/cuda/nvrtc_compiler.hpp"

#include <exception>
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

}  // namespace

int main() {
  using flight4s::cuda::NvrtcCompileRequest;
  const flight4s::cuda::NvrtcCompiler compiler;

  const auto valid = compiler.compile(
      NvrtcCompileRequest{
          R"(
extern "C" __global__ void add_one(float* values) {
  const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  values[index] += 1.0f;
}
)",
          "valid_kernel.cu",
          {
              "--std=c++20",
              "--gpu-architecture=compute_80",
          },
      });

  expect(valid.succeeded(), "valid CUDA source should compile");
  expect(valid.result_code == 0, "NVRTC success code should be zero");
  expect(
      valid.result_name == "NVRTC_SUCCESS",
      "NVRTC success name should be retained");
  expect(!valid.ptx.empty(), "valid CUDA source should produce PTX");
  expect(
      std::string(valid.ptx.begin(), valid.ptx.end())
              .find(".entry add_one") != std::string::npos,
      "PTX should retain the extern C kernel entry name");
  expect(
      valid.version_major >= 12,
      "the configured toolkit should report NVRTC 12 or newer");
  expect(
      valid.version_minor >= 0,
      "NVRTC minor version should not be negative");

  const auto invalid = compiler.compile(
      NvrtcCompileRequest{
          R"(
extern "C" __global__ void broken( {
}
)",
          "invalid_kernel.cu",
          {
              "--std=c++20",
              "--gpu-architecture=compute_80",
          },
      });

  expect(!invalid.succeeded(), "invalid CUDA source should fail");
  expect(
      invalid.result_name == "NVRTC_ERROR_COMPILATION",
      "compile failure should retain the NVRTC result name");
  expect(invalid.ptx.empty(), "failed compilation should not return PTX");
  expect(
      !invalid.compile_log.empty(),
      "failed compilation should preserve the NVRTC log");
  expect(
      invalid.compile_log.find("invalid_kernel.cu") != std::string::npos,
      "the compile log should preserve the logical program name");

  bool rejected_empty_source = false;
  try {
    static_cast<void>(compiler.compile(
        NvrtcCompileRequest{
            "",
            "empty.cu",
            {"--std=c++20"},
        }));
  } catch (const std::invalid_argument&) {
    rejected_empty_source = true;
  }
  expect(rejected_empty_source, "empty CUDA source should be rejected");

  bool rejected_null_option = false;
  std::string null_option = "--std=c++20";
  null_option.push_back('\0');
  null_option += "ignored";
  try {
    static_cast<void>(compiler.compile(
        NvrtcCompileRequest{
            "extern \"C\" __global__ void empty() {}",
            "null_option.cu",
            {null_option},
        }));
  } catch (const std::invalid_argument&) {
    rejected_null_option = true;
  }
  expect(
      rejected_null_option,
      "NVRTC options containing null bytes should be rejected");

  if (failures != 0) {
    std::cerr << failures << " test assertion(s) failed\n";
    return 1;
  }
  std::cout << "NVRTC compiler tests passed\n";
  return 0;
}
