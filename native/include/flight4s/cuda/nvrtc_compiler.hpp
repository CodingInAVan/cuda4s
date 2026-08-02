#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace flight4s::cuda {

struct NvrtcCompileRequest {
  std::string source;
  std::string program_name;
  std::vector<std::string> options;
};

struct NvrtcVersionResult {
  std::int32_t result_code;
  std::string result_name;
  std::int32_t version_major;
  std::int32_t version_minor;

  [[nodiscard]] bool succeeded() const noexcept {
    return result_code == 0;
  }
};

struct NvrtcCompilation {
  std::int32_t result_code;
  std::string result_name;
  std::vector<std::uint8_t> ptx;
  std::string compile_log;
  std::int32_t version_major;
  std::int32_t version_minor;

  [[nodiscard]] bool succeeded() const noexcept {
    return result_code == 0;
  }
};

class NvrtcCompiler final {
 public:
  [[nodiscard]] NvrtcVersionResult version() const;

  [[nodiscard]] NvrtcCompilation compile(
      const NvrtcCompileRequest& request) const;
};

}  // namespace flight4s::cuda
