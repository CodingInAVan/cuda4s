#include <jni.h>

#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "flight4s/cuda/nvrtc_compiler.hpp"
#include "jni_environment.hpp"

namespace {

class NvrtcJniAdapter final {
 public:
  explicit NvrtcJniAdapter(JNIEnv* environment) noexcept
      : environment_(environment) {}

  [[nodiscard]] jobject compile(
      jbyteArray source_utf8,
      jbyteArray program_name_utf8,
      jobjectArray options_utf8) {
    auto source = read_utf8_bytes(source_utf8, "CUDA source");
    if (has_exception()) {
      return nullptr;
    }
    auto program_name =
        read_utf8_bytes(program_name_utf8, "program name");
    if (has_exception()) {
      return nullptr;
    }
    auto options = read_options(options_utf8);
    if (has_exception()) {
      return nullptr;
    }

    const auto result = compiler_.compile(
        {std::move(source), std::move(program_name), std::move(options)});
    return to_java_result(result);
  }

 private:
  [[nodiscard]] bool has_exception() const noexcept {
    return environment_->ExceptionCheck() == JNI_TRUE;
  }

  [[nodiscard]] std::string read_utf8_bytes(
      jbyteArray bytes,
      const char* description) const {
    if (bytes == nullptr) {
      throw std::invalid_argument(
          std::string(description) + " must not be null");
    }

    const jsize size = environment_->GetArrayLength(bytes);
    std::string value(static_cast<std::size_t>(size), '\0');
    if (size > 0) {
      environment_->GetByteArrayRegion(
          bytes,
          0,
          size,
          reinterpret_cast<jbyte*>(value.data()));
    }
    return value;
  }

  [[nodiscard]] std::vector<std::string> read_options(
      jobjectArray options) const {
    if (options == nullptr) {
      throw std::invalid_argument(
          "NVRTC options must not be null");
    }

    const jsize count = environment_->GetArrayLength(options);
    std::vector<std::string> values;
    values.reserve(static_cast<std::size_t>(count));
    for (jsize index = 0; index < count; ++index) {
      auto option = static_cast<jbyteArray>(
          environment_->GetObjectArrayElement(options, index));
      if (has_exception()) {
        return {};
      }
      if (option == nullptr) {
        throw std::invalid_argument(
            "NVRTC options must not contain null");
      }

      values.push_back(read_utf8_bytes(option, "NVRTC option"));
      environment_->DeleteLocalRef(option);
      if (has_exception()) {
        return {};
      }
    }
    return values;
  }

  [[nodiscard]] jbyteArray to_byte_array(
      const std::uint8_t* bytes,
      std::size_t size) const {
    if (size > static_cast<std::size_t>(
                   std::numeric_limits<jsize>::max())) {
      throw std::overflow_error(
          "native compilation artifact exceeds the JVM array limit");
    }

    auto result =
        environment_->NewByteArray(static_cast<jsize>(size));
    if (result != nullptr && size > 0) {
      environment_->SetByteArrayRegion(
          result,
          0,
          static_cast<jsize>(size),
          reinterpret_cast<const jbyte*>(bytes));
    }
    return result;
  }

  [[nodiscard]] jbyteArray to_byte_array(
      const std::string& value) const {
    return to_byte_array(
        reinterpret_cast<const std::uint8_t*>(value.data()),
        value.size());
  }

  [[nodiscard]] jobject to_java_result(
      const flight4s::cuda::NvrtcCompilation& result) const {
    auto ptx = to_byte_array(result.ptx.data(), result.ptx.size());
    if (ptx == nullptr || has_exception()) {
      return nullptr;
    }
    auto log = to_byte_array(result.compile_log);
    if (log == nullptr || has_exception()) {
      environment_->DeleteLocalRef(ptx);
      return nullptr;
    }
    auto result_name = to_byte_array(result.result_name);
    if (result_name == nullptr || has_exception()) {
      environment_->DeleteLocalRef(log);
      environment_->DeleteLocalRef(ptx);
      return nullptr;
    }

    jclass result_class = environment_->FindClass(
        "flight4s/runtime/cuda/internal/NativeNvrtcResult");
    if (result_class == nullptr) {
      environment_->DeleteLocalRef(result_name);
      environment_->DeleteLocalRef(log);
      environment_->DeleteLocalRef(ptx);
      return nullptr;
    }
    const jmethodID constructor = environment_->GetMethodID(
        result_class,
        "<init>",
        "([B[B[BIII)V");
    if (constructor == nullptr) {
      environment_->DeleteLocalRef(result_class);
      environment_->DeleteLocalRef(result_name);
      environment_->DeleteLocalRef(log);
      environment_->DeleteLocalRef(ptx);
      return nullptr;
    }

    jobject java_result = environment_->NewObject(
        result_class,
        constructor,
        ptx,
        log,
        result_name,
        static_cast<jint>(result.result_code),
        static_cast<jint>(result.version_major),
        static_cast<jint>(result.version_minor));
    environment_->DeleteLocalRef(result_class);
    environment_->DeleteLocalRef(result_name);
    environment_->DeleteLocalRef(log);
    environment_->DeleteLocalRef(ptx);
    return java_result;
  }

  JNIEnv* environment_;
  flight4s::cuda::NvrtcCompiler compiler_;
};

}  // namespace

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_compileCuda(
    JNIEnv* environment,
    jclass,
    jbyteArray source_utf8,
    jbyteArray program_name_utf8,
    jobjectArray options_utf8) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return NvrtcJniAdapter(environment).compile(
        source_utf8,
        program_name_utf8,
        options_utf8);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}
