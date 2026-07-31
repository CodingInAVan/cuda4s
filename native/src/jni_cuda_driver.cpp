#include <jni.h>

#include <cstdint>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "flight4s/cuda/cuda_driver.hpp"
#include "jni_cuda_result.hpp"
#include "jni_environment.hpp"

namespace {

template <typename Handle>
Handle from_java_handle(jlong handle) {
  return reinterpret_cast<Handle>(
      static_cast<std::uintptr_t>(handle));
}

template <typename Handle>
jlong to_java_handle(Handle handle) {
  return static_cast<jlong>(
      reinterpret_cast<std::uintptr_t>(handle));
}

class CudaDriverJniAdapter final {
 public:
  explicit CudaDriverJniAdapter(JNIEnv* environment) noexcept
      : environment_(environment),
        results_(environment) {}

  [[nodiscard]] jobject retain_primary_context(
      jint device_ordinal) const {
    const auto result =
        driver_.retain_primary_context(device_ordinal);
    return results_.context_result(result);
  }

  [[nodiscard]] jobject release_primary_context(
      jint device_ordinal) const {
    const auto status =
        driver_.release_primary_context(device_ordinal);
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject load_ptx(
      jlong context_handle,
      jbyteArray ptx) const {
    const auto bytes = read_bytes(ptx, "PTX");
    if (has_exception()) {
      return nullptr;
    }
    const auto result = driver_.load_ptx(
        from_java_handle<CUcontext>(context_handle),
        bytes);
    return results_.driver_result(
        result.status,
        to_java_handle(result.module));
  }

  [[nodiscard]] jobject unload_module(
      jlong context_handle,
      jlong module_handle) const {
    const auto status = driver_.unload_module(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<CUmodule>(module_handle));
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject resolve_function(
      jlong context_handle,
      jlong module_handle,
      jbyteArray function_name_utf8) const {
    const auto name_bytes =
        read_bytes(function_name_utf8, "CUDA function name");
    if (has_exception()) {
      return nullptr;
    }
    const std::string name(name_bytes.begin(), name_bytes.end());
    const auto result = driver_.resolve_function(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<CUmodule>(module_handle),
        name);
    return results_.driver_result(
        result.status,
        to_java_handle(result.function));
  }

 private:
  [[nodiscard]] bool has_exception() const noexcept {
    return environment_->ExceptionCheck() == JNI_TRUE;
  }

  [[nodiscard]] std::vector<std::uint8_t> read_bytes(
      jbyteArray bytes,
      const char* description) const {
    if (bytes == nullptr) {
      throw std::invalid_argument(
          std::string(description) + " must not be null");
    }

    const jsize size = environment_->GetArrayLength(bytes);
    std::vector<std::uint8_t> value(
        static_cast<std::size_t>(size));
    if (size > 0) {
      environment_->GetByteArrayRegion(
          bytes,
          0,
          size,
          reinterpret_cast<jbyte*>(value.data()));
    }
    return value;
  }

  JNIEnv* environment_;
  flight4s::jni::CudaJniResultFactory results_;
  flight4s::cuda::CudaDriver driver_;
};

}  // namespace

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_retainPrimaryContext(
    JNIEnv* environment,
    jclass,
    jint device_ordinal) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .retain_primary_context(device_ordinal);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_releasePrimaryContext(
    JNIEnv* environment,
    jclass,
    jint device_ordinal) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .release_primary_context(device_ordinal);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_loadPtx(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jbyteArray ptx) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .load_ptx(context_handle, ptx);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_unloadModule(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong module_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .unload_module(context_handle, module_handle);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_resolveFunction(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong module_handle,
    jbyteArray function_name_utf8) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .resolve_function(
            context_handle,
            module_handle,
            function_name_utf8);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}
