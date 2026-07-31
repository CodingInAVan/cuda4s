#include <jni.h>

#include <cstdint>
#include <exception>
#include <new>
#include <stdexcept>
#include <vector>

#include "flight4s/cuda/cuda_launcher.hpp"
#include "jni_direct_buffer.hpp"
#include "jni_cuda_result.hpp"
#include "jni_environment.hpp"

namespace {

class CudaLaunchJniAdapter final {
 public:
  explicit CudaLaunchJniAdapter(JNIEnv* environment) noexcept
      : environment_(environment),
        results_(environment) {}

  [[nodiscard]] jobject launch(
      jlong context_handle,
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
    const auto storage = flight4s::jni::DirectBufferView::require_exact(
        environment_,
        argument_storage,
        "argument storage");
    if (has_exception()) {
      return nullptr;
    }

    auto offsets = read_offsets(argument_offsets);
    if (has_exception()) {
      return nullptr;
    }
    auto descriptor_codes =
        read_descriptor_codes(argument_descriptor_codes);
    if (has_exception()) {
      return nullptr;
    }

    const flight4s::cuda::CudaLaunchRequest request{
        reinterpret_cast<CUcontext>(
            static_cast<std::uintptr_t>(context_handle)),
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
            storage.data(),
            storage.size_bytes(),
            offsets.data(),
            offsets.size(),
            descriptor_codes.data(),
            descriptor_codes.size(),
        },
    };

    return results_.driver_result(launcher_.launch(request));
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
  flight4s::jni::CudaJniResultFactory results_;
  flight4s::cuda::CudaLauncher launcher_;
};

}  // namespace

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_launchKernel(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
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
        context_handle,
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
  return nullptr;
}
