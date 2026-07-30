#include "flight4s/cuda/nvrtc_compiler.hpp"

#include <nvrtc.h>

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace flight4s::cuda {
namespace {

std::string without_trailing_null(std::vector<char> bytes);

class NvrtcProgram final {
 public:
  explicit NvrtcProgram(nvrtcProgram program) noexcept
      : program_(program) {}

  NvrtcProgram(const NvrtcProgram&) = delete;
  NvrtcProgram& operator=(const NvrtcProgram&) = delete;
  NvrtcProgram(NvrtcProgram&&) = delete;
  NvrtcProgram& operator=(NvrtcProgram&&) = delete;

  ~NvrtcProgram() {
    if (program_ != nullptr) {
      nvrtcDestroyProgram(&program_);
    }
  }

  [[nodiscard]] nvrtcResult compile(
      const std::vector<std::string>& options) const {
    std::vector<const char*> option_pointers;
    option_pointers.reserve(options.size());
    std::transform(
        options.begin(),
        options.end(),
        std::back_inserter(option_pointers),
        [](const std::string& option) {
          return option.c_str();
        });

    return nvrtcCompileProgram(
        program_,
        static_cast<int>(option_pointers.size()),
        option_pointers.empty() ? nullptr : option_pointers.data());
  }

  [[nodiscard]] nvrtcResult read_log(std::string& output) const {
    std::size_t log_size = 0;
    const auto size_result =
        nvrtcGetProgramLogSize(program_, &log_size);
    if (size_result != NVRTC_SUCCESS) {
      return size_result;
    }
    if (log_size == 0) {
      output.clear();
      return NVRTC_SUCCESS;
    }

    std::vector<char> log(log_size);
    const auto log_result =
        nvrtcGetProgramLog(program_, log.data());
    if (log_result == NVRTC_SUCCESS) {
      output = without_trailing_null(std::move(log));
    }
    return log_result;
  }

  [[nodiscard]] nvrtcResult read_ptx(
      std::vector<std::uint8_t>& output) const {
    std::size_t ptx_size = 0;
    const auto size_result = nvrtcGetPTXSize(program_, &ptx_size);
    if (size_result != NVRTC_SUCCESS) {
      return size_result;
    }
    if (ptx_size == 0) {
      throw std::runtime_error(
          "successful NVRTC compilation returned empty PTX");
    }

    std::vector<char> ptx(ptx_size);
    const auto ptx_result = nvrtcGetPTX(program_, ptx.data());
    if (ptx_result != NVRTC_SUCCESS) {
      return ptx_result;
    }
    if (!ptx.empty() && ptx.back() == '\0') {
      ptx.pop_back();
    }
    output.assign(ptx.begin(), ptx.end());
    if (output.empty()) {
      throw std::runtime_error(
          "successful NVRTC compilation returned only a null terminator");
    }
    return NVRTC_SUCCESS;
  }

 private:
  nvrtcProgram program_;
};

void require_c_string(
    std::string_view value,
    const char* description,
    bool allow_empty) {
  if (!allow_empty && value.empty()) {
    throw std::invalid_argument(
        std::string(description) + " must not be empty");
  }
  if (value.find('\0') != std::string_view::npos) {
    throw std::invalid_argument(
        std::string(description) + " must not contain a null byte");
  }
}

void set_result(
    NvrtcCompilation& result,
    nvrtcResult code) {
  result.result_code = static_cast<std::int32_t>(code);
  const char* name = nvrtcGetErrorString(code);
  result.result_name =
      name == nullptr ? "NVRTC_ERROR_UNKNOWN" : name;
}

std::string without_trailing_null(std::vector<char> bytes) {
  if (!bytes.empty() && bytes.back() == '\0') {
    bytes.pop_back();
  }
  return std::string(bytes.begin(), bytes.end());
}

}  // namespace

NvrtcCompilation NvrtcCompiler::compile(
    const NvrtcCompileRequest& request) const {
  require_c_string(request.source, "CUDA source", false);
  require_c_string(request.program_name, "program name", false);
  for (const auto& option : request.options) {
    require_c_string(option, "NVRTC option", false);
  }

  NvrtcCompilation result{
      static_cast<std::int32_t>(NVRTC_SUCCESS),
      nvrtcGetErrorString(NVRTC_SUCCESS),
      {},
      {},
      0,
      0,
  };

  int version_major = 0;
  int version_minor = 0;
  const auto version_result =
      nvrtcVersion(&version_major, &version_minor);
  if (version_result != NVRTC_SUCCESS) {
    set_result(result, version_result);
    return result;
  }
  result.version_major = version_major;
  result.version_minor = version_minor;

  nvrtcProgram program = nullptr;
  const auto create_result = nvrtcCreateProgram(
      &program,
      request.source.c_str(),
      request.program_name.c_str(),
      0,
      nullptr,
      nullptr);
  if (create_result != NVRTC_SUCCESS) {
    set_result(result, create_result);
    return result;
  }
  const NvrtcProgram nvrtc_program(program);

  const auto compile_result =
      nvrtc_program.compile(request.options);
  const auto log_result =
      nvrtc_program.read_log(result.compile_log);
  if (log_result != NVRTC_SUCCESS) {
    set_result(result, log_result);
    return result;
  }

  if (compile_result != NVRTC_SUCCESS) {
    set_result(result, compile_result);
    return result;
  }

  const auto ptx_result = nvrtc_program.read_ptx(result.ptx);
  if (ptx_result != NVRTC_SUCCESS) {
    set_result(result, ptx_result);
    return result;
  }

  set_result(result, NVRTC_SUCCESS);
  return result;
}

}  // namespace flight4s::cuda
