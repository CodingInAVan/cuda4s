#pragma once

#include <jni.h>

#include <string>

#include "flight4s/cuda/cuda_driver.hpp"

namespace flight4s::jni {

class CudaJniResultFactory final {
 public:
  explicit CudaJniResultFactory(JNIEnv* environment) noexcept;

  [[nodiscard]] jobject context_result(
      const cuda::CudaContextResult& result) const;

  [[nodiscard]] jobject driver_result(
      const cuda::CudaDriverStatus& status,
      jlong handle = 0) const;

 private:
  [[nodiscard]] bool has_exception() const noexcept;
  [[nodiscard]] jbyteArray byte_array(const std::string& value) const;

  JNIEnv* environment_;
};

}  // namespace flight4s::jni
