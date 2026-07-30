#pragma once

#include <jni.h>

#include <string>

namespace flight4s::jni {

class JniEnvironment final {
 public:
  explicit JniEnvironment(JNIEnv* environment) noexcept;

  void throw_illegal_argument(const std::string& message) const;
  void throw_out_of_memory(const std::string& message) const;
  void throw_runtime_exception(const std::string& message) const;

 private:
  void throw_java(
      const char* class_name,
      const std::string& message) const;

  JNIEnv* environment_;
};

}  // namespace flight4s::jni
