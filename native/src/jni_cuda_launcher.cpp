#include <jni.h>

#include <cstdint>
#include <exception>
#include <new>
#include <stdexcept>
#include <vector>

#include "flight4s/cuda/cuda_launcher.hpp"
#include "jni_environment.hpp"

namespace {

class CudaLaunchJniAdapter final {
 public:
  explicit CudaLaunchJniAdapter(JNIEnv* environment) noexcept
      : environment_(environment) {}

  [[nodiscard]] jint launch(
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
      jbyteArray argument_descriptor_codes) const {
    require_arguments(
        argument_storage,
        argument_offsets,
        argument_descriptor_codes);
    validate_exact_buffer_view(argument_storage);
    if (has_exception()) {
      return 0;
    }

    void* storage =
        environment_->GetDirectBufferAddress(argument_storage);
    const jlong storage_size =
        environment_->GetDirectBufferCapacity(argument_storage);
    auto offsets = read_offsets(argument_offsets);
    if (has_exception()) {
      return 0;
    }
    auto descriptor_codes =
        read_descriptor_codes(argument_descriptor_codes);
    if (has_exception()) {
      return 0;
    }

    const flight4s::cuda::CudaLaunchRequest request{
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

    return static_cast<jint>(launcher_.launch(request));
  }

 private:
  static void require_arguments(
      jobject argument_storage,
      jintArray argument_offsets,
      jbyteArray argument_descriptor_codes) {
    if (argument_storage == nullptr) {
      throw std::invalid_argument(
          "argument storage buffer must not be null");
    }
    if (argument_offsets == nullptr ||
        argument_descriptor_codes == nullptr) {
      throw std::invalid_argument(
          "argument metadata arrays must not be null");
    }
  }

  [[nodiscard]] bool has_exception() const noexcept {
    return environment_->ExceptionCheck() == JNI_TRUE;
  }

  void validate_exact_buffer_view(jobject buffer) const {
    jclass buffer_class = environment_->GetObjectClass(buffer);
    if (buffer_class == nullptr) {
      return;
    }
    const jmethodID position_method =
        environment_->GetMethodID(buffer_class, "position", "()I");
    const jmethodID limit_method =
        environment_->GetMethodID(buffer_class, "limit", "()I");
    const jmethodID capacity_method =
        environment_->GetMethodID(buffer_class, "capacity", "()I");
    if (position_method == nullptr || limit_method == nullptr ||
        capacity_method == nullptr) {
      environment_->DeleteLocalRef(buffer_class);
      return;
    }

    const jint position =
        environment_->CallIntMethod(buffer, position_method);
    const jint limit =
        environment_->CallIntMethod(buffer, limit_method);
    const jint capacity =
        environment_->CallIntMethod(buffer, capacity_method);
    environment_->DeleteLocalRef(buffer_class);
    if (has_exception()) {
      return;
    }
    if (position != 0 || limit != capacity) {
      throw std::invalid_argument(
          "argument storage must have position 0 and limit equal to capacity");
    }
  }

  [[nodiscard]] std::vector<std::int32_t> read_offsets(
      jintArray offsets) const {
    const auto count = environment_->GetArrayLength(offsets);
    std::vector<jint> raw_values(static_cast<std::size_t>(count));
    if (count > 0) {
      environment_->GetIntArrayRegion(
          offsets,
          0,
          count,
          raw_values.data());
    }
    return std::vector<std::int32_t>(
        raw_values.begin(),
        raw_values.end());
  }

  [[nodiscard]] std::vector<std::int8_t> read_descriptor_codes(
      jbyteArray descriptor_codes) const {
    const auto count = environment_->GetArrayLength(descriptor_codes);
    std::vector<jbyte> raw_values(static_cast<std::size_t>(count));
    if (count > 0) {
      environment_->GetByteArrayRegion(
          descriptor_codes,
          0,
          count,
          raw_values.data());
    }
    return std::vector<std::int8_t>(
        raw_values.begin(),
        raw_values.end());
  }

  JNIEnv* environment_;
  flight4s::cuda::CudaLauncher launcher_;
};

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
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaLaunchJniAdapter(environment).launch(
        function_handle,
        stream_handle,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        dynamic_shared_memory_bytes,
        uses_cluster,
        cluster_x,
        cluster_y,
        cluster_z,
        argument_storage,
        argument_offsets,
        argument_descriptor_codes);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return 0;
}
