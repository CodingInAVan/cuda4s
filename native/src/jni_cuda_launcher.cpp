#include <jni.h>

#include <cstdint>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "flight4s/cuda/cuda_launcher.hpp"

namespace {

void throw_java(
    JNIEnv* environment,
    const char* class_name,
    const std::string& message) {
  jclass exception_class = environment->FindClass(class_name);
  if (exception_class != nullptr) {
    environment->ThrowNew(exception_class, message.c_str());
  }
}

void validate_exact_buffer_view(
    JNIEnv* environment,
    jobject buffer) {
  jclass buffer_class = environment->GetObjectClass(buffer);
  if (buffer_class == nullptr) {
    return;
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
    return;
  }

  const jint position =
      environment->CallIntMethod(buffer, position_method);
  const jint limit = environment->CallIntMethod(buffer, limit_method);
  const jint capacity =
      environment->CallIntMethod(buffer, capacity_method);
  environment->DeleteLocalRef(buffer_class);
  if (environment->ExceptionCheck()) {
    return;
  }
  if (position != 0 || limit != capacity) {
    throw std::invalid_argument(
        "argument storage must have position 0 and limit equal to capacity");
  }
}

std::vector<std::int32_t> read_offsets(
    JNIEnv* environment,
    jintArray offsets) {
  const auto count = environment->GetArrayLength(offsets);
  std::vector<jint> raw_values(static_cast<std::size_t>(count));
  if (count > 0) {
    environment->GetIntArrayRegion(
        offsets,
        0,
        count,
        raw_values.data());
  }
  return std::vector<std::int32_t>(
      raw_values.begin(),
      raw_values.end());
}

std::vector<std::int8_t> read_descriptor_codes(
    JNIEnv* environment,
    jbyteArray descriptor_codes) {
  const auto count = environment->GetArrayLength(descriptor_codes);
  std::vector<jbyte> raw_values(static_cast<std::size_t>(count));
  if (count > 0) {
    environment->GetByteArrayRegion(
        descriptor_codes,
        0,
        count,
        raw_values.data());
  }
  return std::vector<std::int8_t>(
      raw_values.begin(),
      raw_values.end());
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_launchKernel(
    JNIEnv* environment,
    jclass,
    jlong function_handle,
    jlong stream_handle,
    jint grid_x,
    jint grid_y,
    jint grid_z,
    jint block_x,
    jint block_y,
    jint block_z,
    jint dynamic_shared_memory_bytes,
    jboolean uses_cluster,
    jint cluster_x,
    jint cluster_y,
    jint cluster_z,
    jobject argument_storage,
    jintArray argument_offsets,
    jbyteArray argument_descriptor_codes) {
  try {
    if (argument_storage == nullptr) {
      throw std::invalid_argument(
          "argument storage buffer must not be null");
    }
    if (argument_offsets == nullptr ||
        argument_descriptor_codes == nullptr) {
      throw std::invalid_argument(
          "argument metadata arrays must not be null");
    }

    validate_exact_buffer_view(environment, argument_storage);
    if (environment->ExceptionCheck()) {
      return 0;
    }

    void* storage =
        environment->GetDirectBufferAddress(argument_storage);
    const jlong storage_size =
        environment->GetDirectBufferCapacity(argument_storage);
    auto offsets = read_offsets(environment, argument_offsets);
    if (environment->ExceptionCheck()) {
      return 0;
    }
    auto descriptor_codes =
        read_descriptor_codes(environment, argument_descriptor_codes);
    if (environment->ExceptionCheck()) {
      return 0;
    }

    flight4s::cuda::CudaLaunchRequest request{
        reinterpret_cast<CUfunction>(
            static_cast<std::uintptr_t>(function_handle)),
        reinterpret_cast<CUstream>(
            static_cast<std::uintptr_t>(stream_handle)),
        {
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            dynamic_shared_memory_bytes,
            uses_cluster == JNI_TRUE,
            cluster_x,
            cluster_y,
            cluster_z,
        },
        {
            storage,
            storage_size,
            offsets.data(),
            offsets.size(),
            descriptor_codes.data(),
            descriptor_codes.size(),
        },
    };

    return static_cast<jint>(flight4s::cuda::launch_kernel(request));
  } catch (const std::invalid_argument& error) {
    throw_java(
        environment,
        "java/lang/IllegalArgumentException",
        error.what());
  } catch (const std::bad_alloc& error) {
    throw_java(
        environment,
        "java/lang/OutOfMemoryError",
        error.what());
  } catch (const std::exception& error) {
    throw_java(
        environment,
        "java/lang/RuntimeException",
        error.what());
  }
  return 0;
}
