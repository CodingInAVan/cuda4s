#include "jni_cuda_result.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace flight4s::jni {
namespace {

template <typename Handle>
jlong to_java_handle(Handle handle) {
  return static_cast<jlong>(
      reinterpret_cast<std::uintptr_t>(handle));
}

}  // namespace

CudaJniResultFactory::CudaJniResultFactory(
    JNIEnv* environment) noexcept
    : environment_(environment) {}

bool CudaJniResultFactory::has_exception() const noexcept {
  return environment_->ExceptionCheck() == JNI_TRUE;
}

jbyteArray CudaJniResultFactory::byte_array(
    const std::string& value) const {
  if (value.size() >
      static_cast<std::size_t>(std::numeric_limits<jsize>::max())) {
    throw std::overflow_error(
        "CUDA Driver text exceeds the JVM array limit");
  }

  auto result =
      environment_->NewByteArray(static_cast<jsize>(value.size()));
  if (result != nullptr && !value.empty()) {
    environment_->SetByteArrayRegion(
        result,
        0,
        static_cast<jsize>(value.size()),
        reinterpret_cast<const jbyte*>(value.data()));
  }
  return result;
}

jobject CudaJniResultFactory::context_result(
    const cuda::CudaContextResult& result) const {
  auto result_name = byte_array(result.status.result_name);
  if (result_name == nullptr || has_exception()) {
    return nullptr;
  }
  auto result_description =
      byte_array(result.status.result_description);
  if (result_description == nullptr || has_exception()) {
    environment_->DeleteLocalRef(result_name);
    return nullptr;
  }

  jclass result_class = environment_->FindClass(
      "flight4s/runtime/cuda/internal/NativeCudaContextResult");
  if (result_class == nullptr) {
    environment_->DeleteLocalRef(result_description);
    environment_->DeleteLocalRef(result_name);
    return nullptr;
  }
  const jmethodID constructor = environment_->GetMethodID(
      result_class,
      "<init>",
      "(JIIII[B[B)V");
  if (constructor == nullptr) {
    environment_->DeleteLocalRef(result_class);
    environment_->DeleteLocalRef(result_description);
    environment_->DeleteLocalRef(result_name);
    return nullptr;
  }

  jobject java_result = environment_->NewObject(
      result_class,
      constructor,
      to_java_handle(result.context),
      static_cast<jint>(result.device_ordinal),
      static_cast<jint>(result.compute_capability_major),
      static_cast<jint>(result.compute_capability_minor),
      static_cast<jint>(result.status.result_code),
      result_name,
      result_description);
  environment_->DeleteLocalRef(result_class);
  environment_->DeleteLocalRef(result_description);
  environment_->DeleteLocalRef(result_name);
  return java_result;
}

jobject CudaJniResultFactory::driver_result(
    const cuda::CudaDriverStatus& status,
    jlong handle) const {
  auto result_name = byte_array(status.result_name);
  if (result_name == nullptr || has_exception()) {
    return nullptr;
  }
  auto result_description = byte_array(status.result_description);
  if (result_description == nullptr || has_exception()) {
    environment_->DeleteLocalRef(result_name);
    return nullptr;
  }
  auto info_log = byte_array(status.info_log);
  if (info_log == nullptr || has_exception()) {
    environment_->DeleteLocalRef(result_description);
    environment_->DeleteLocalRef(result_name);
    return nullptr;
  }
  auto error_log = byte_array(status.error_log);
  if (error_log == nullptr || has_exception()) {
    environment_->DeleteLocalRef(info_log);
    environment_->DeleteLocalRef(result_description);
    environment_->DeleteLocalRef(result_name);
    return nullptr;
  }

  jclass result_class = environment_->FindClass(
      "flight4s/runtime/cuda/internal/NativeCudaDriverResult");
  if (result_class == nullptr) {
    environment_->DeleteLocalRef(error_log);
    environment_->DeleteLocalRef(info_log);
    environment_->DeleteLocalRef(result_description);
    environment_->DeleteLocalRef(result_name);
    return nullptr;
  }
  const jmethodID constructor = environment_->GetMethodID(
      result_class,
      "<init>",
      "(JI[B[B[B[B)V");
  if (constructor == nullptr) {
    environment_->DeleteLocalRef(result_class);
    environment_->DeleteLocalRef(error_log);
    environment_->DeleteLocalRef(info_log);
    environment_->DeleteLocalRef(result_description);
    environment_->DeleteLocalRef(result_name);
    return nullptr;
  }

  jobject java_result = environment_->NewObject(
      result_class,
      constructor,
      handle,
      static_cast<jint>(status.result_code),
      result_name,
      result_description,
      info_log,
      error_log);
  environment_->DeleteLocalRef(result_class);
  environment_->DeleteLocalRef(error_log);
  environment_->DeleteLocalRef(info_log);
  environment_->DeleteLocalRef(result_description);
  environment_->DeleteLocalRef(result_name);
  return java_result;
}

jobject CudaJniResultFactory::event_query_result(
    const cuda::CudaEventQueryResult& result) const {
  auto result_name = byte_array(result.status.result_name);
  if (result_name == nullptr || has_exception()) {
    return nullptr;
  }
  auto result_description = byte_array(result.status.result_description);
  if (result_description == nullptr || has_exception()) {
    environment_->DeleteLocalRef(result_name);
    return nullptr;
  }
  auto info_log = byte_array(result.status.info_log);
  if (info_log == nullptr || has_exception()) {
    environment_->DeleteLocalRef(result_description);
    environment_->DeleteLocalRef(result_name);
    return nullptr;
  }
  auto error_log = byte_array(result.status.error_log);
  if (error_log == nullptr || has_exception()) {
    environment_->DeleteLocalRef(info_log);
    environment_->DeleteLocalRef(result_description);
    environment_->DeleteLocalRef(result_name);
    return nullptr;
  }

  jclass result_class = environment_->FindClass(
      "flight4s/runtime/cuda/internal/NativeCudaEventQueryResult");
  if (result_class == nullptr) {
    environment_->DeleteLocalRef(error_log);
    environment_->DeleteLocalRef(info_log);
    environment_->DeleteLocalRef(result_description);
    environment_->DeleteLocalRef(result_name);
    return nullptr;
  }
  const jmethodID constructor = environment_->GetMethodID(
      result_class,
      "<init>",
      "(ZI[B[B[B[B)V");
  if (constructor == nullptr) {
    environment_->DeleteLocalRef(result_class);
    environment_->DeleteLocalRef(error_log);
    environment_->DeleteLocalRef(info_log);
    environment_->DeleteLocalRef(result_description);
    environment_->DeleteLocalRef(result_name);
    return nullptr;
  }

  jobject java_result = environment_->NewObject(
      result_class,
      constructor,
      result.complete ? JNI_TRUE : JNI_FALSE,
      static_cast<jint>(result.status.result_code),
      result_name,
      result_description,
      info_log,
      error_log);
  environment_->DeleteLocalRef(result_class);
  environment_->DeleteLocalRef(error_log);
  environment_->DeleteLocalRef(info_log);
  environment_->DeleteLocalRef(result_description);
  environment_->DeleteLocalRef(result_name);
  return java_result;
}

jobject CudaJniResultFactory::pinned_memory_result(
    const cuda::CudaPinnedMemoryResult& result,
    jlong size_bytes) const {
  auto result_name = byte_array(result.status.result_name);
  if (result_name == nullptr || has_exception()) {
    return nullptr;
  }
  auto result_description = byte_array(result.status.result_description);
  if (result_description == nullptr || has_exception()) {
    environment_->DeleteLocalRef(result_name);
    return nullptr;
  }
  auto info_log = byte_array(result.status.info_log);
  if (info_log == nullptr || has_exception()) {
    environment_->DeleteLocalRef(result_description);
    environment_->DeleteLocalRef(result_name);
    return nullptr;
  }
  auto error_log = byte_array(result.status.error_log);
  if (error_log == nullptr || has_exception()) {
    environment_->DeleteLocalRef(info_log);
    environment_->DeleteLocalRef(result_description);
    environment_->DeleteLocalRef(result_name);
    return nullptr;
  }

  jobject storage = nullptr;
  if (result.status.succeeded()) {
    if (result.address == nullptr) {
      environment_->DeleteLocalRef(error_log);
      environment_->DeleteLocalRef(info_log);
      environment_->DeleteLocalRef(result_description);
      environment_->DeleteLocalRef(result_name);
      throw std::logic_error(
          "successful pinned allocation returned a null address");
    }
    storage = environment_->NewDirectByteBuffer(
        result.address,
        size_bytes);
    if (storage == nullptr || has_exception()) {
      environment_->DeleteLocalRef(error_log);
      environment_->DeleteLocalRef(info_log);
      environment_->DeleteLocalRef(result_description);
      environment_->DeleteLocalRef(result_name);
      return nullptr;
    }
  }

  jclass result_class = environment_->FindClass(
      "flight4s/runtime/cuda/internal/NativeCudaPinnedMemoryResult");
  if (result_class == nullptr) {
    if (storage != nullptr) {
      environment_->DeleteLocalRef(storage);
    }
    environment_->DeleteLocalRef(error_log);
    environment_->DeleteLocalRef(info_log);
    environment_->DeleteLocalRef(result_description);
    environment_->DeleteLocalRef(result_name);
    return nullptr;
  }
  const jmethodID constructor = environment_->GetMethodID(
      result_class,
      "<init>",
      "(JLjava/nio/ByteBuffer;I[B[B[B[B)V");
  if (constructor == nullptr) {
    environment_->DeleteLocalRef(result_class);
    if (storage != nullptr) {
      environment_->DeleteLocalRef(storage);
    }
    environment_->DeleteLocalRef(error_log);
    environment_->DeleteLocalRef(info_log);
    environment_->DeleteLocalRef(result_description);
    environment_->DeleteLocalRef(result_name);
    return nullptr;
  }

  jobject java_result = environment_->NewObject(
      result_class,
      constructor,
      to_java_handle(result.address),
      storage,
      static_cast<jint>(result.status.result_code),
      result_name,
      result_description,
      info_log,
      error_log);
  environment_->DeleteLocalRef(result_class);
  if (storage != nullptr) {
    environment_->DeleteLocalRef(storage);
  }
  environment_->DeleteLocalRef(error_log);
  environment_->DeleteLocalRef(info_log);
  environment_->DeleteLocalRef(result_description);
  environment_->DeleteLocalRef(result_name);
  return java_result;
}

}  // namespace flight4s::jni
