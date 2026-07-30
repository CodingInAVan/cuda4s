#include "jni_environment.hpp"

namespace flight4s::jni {

JniEnvironment::JniEnvironment(JNIEnv* environment) noexcept
    : environment_(environment) {}

void JniEnvironment::throw_illegal_argument(
    const std::string& message) const {
  throw_java("java/lang/IllegalArgumentException", message);
}

void JniEnvironment::throw_out_of_memory(
    const std::string& message) const {
  throw_java("java/lang/OutOfMemoryError", message);
}

void JniEnvironment::throw_runtime_exception(
    const std::string& message) const {
  throw_java("java/lang/RuntimeException", message);
}

void JniEnvironment::throw_java(
    const char* class_name,
    const std::string& message) const {
  jclass exception_class = environment_->FindClass(class_name);
  if (exception_class != nullptr) {
    environment_->ThrowNew(exception_class, message.c_str());
    environment_->DeleteLocalRef(exception_class);
  }
}

}  // namespace flight4s::jni
