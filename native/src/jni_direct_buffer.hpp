#pragma once

#include <jni.h>

#include <cstdint>

namespace flight4s::jni {

class DirectBufferView final {
 public:
  [[nodiscard]] static DirectBufferView require_exact(
      JNIEnv* environment,
      jobject buffer,
      const char* description);

  [[nodiscard]] void* data() const noexcept;
  [[nodiscard]] std::int64_t size_bytes() const noexcept;

 private:
  DirectBufferView(void* data, std::int64_t size_bytes) noexcept;

  void* data_;
  std::int64_t size_bytes_;
};

}  // namespace flight4s::jni
