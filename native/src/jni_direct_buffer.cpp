#include "jni_direct_buffer.hpp"

#include <stdexcept>
#include <string>

namespace flight4s::jni {

DirectBufferView::DirectBufferView(
    void* data,
    std::int64_t size_bytes) noexcept
    : data_(data),
      size_bytes_(size_bytes) {}

DirectBufferView DirectBufferView::require_exact(
    JNIEnv* environment,
    jobject buffer,
    const char* description) {
  if (buffer == nullptr) {
    throw std::invalid_argument(
        std::string(description) + " must not be null");
  }

  const jlong capacity = environment->GetDirectBufferCapacity(buffer);
  if (capacity < 0) {
    throw std::invalid_argument(
        std::string(description) + " must be a direct buffer");
  }
  void* data = environment->GetDirectBufferAddress(buffer);
  if (data == nullptr && capacity > 0) {
    throw std::invalid_argument(
        std::string(description) + " has no native address");
  }

  jclass buffer_class = environment->GetObjectClass(buffer);
  if (buffer_class == nullptr) {
    return {nullptr, 0};
  }
  const jmethodID position_method =
      environment->GetMethodID(buffer_class, "position", "()I");
  const jmethodID limit_method =
      environment->GetMethodID(buffer_class, "limit", "()I");
  const jmethodID capacity_method =
      environment->GetMethodID(buffer_class, "capacity", "()I");
  if (position_method == nullptr || limit_method == nullptr ||
      capacity_method == nullptr) {
    environment->DeleteLocalRef(buffer_class);
    return {nullptr, 0};
  }

  const jint position =
      environment->CallIntMethod(buffer, position_method);
  const jint limit =
      environment->CallIntMethod(buffer, limit_method);
  const jint java_capacity =
      environment->CallIntMethod(buffer, capacity_method);
  environment->DeleteLocalRef(buffer_class);
  if (environment->ExceptionCheck() == JNI_TRUE) {
    return {nullptr, 0};
  }
  if (position != 0 || limit != java_capacity ||
      static_cast<jlong>(java_capacity) != capacity) {
    throw std::invalid_argument(
        std::string(description) +
        " must have position 0 and limit equal to capacity");
  }

  return {data, static_cast<std::int64_t>(capacity)};
}

void* DirectBufferView::data() const noexcept {
  return data_;
}

std::int64_t DirectBufferView::size_bytes() const noexcept {
  return size_bytes_;
}

}  // namespace flight4s::jni
