#include <atomic>
#include <jni.h>
#include <nvrtc.h>
#include <cuda.h>

#include <string>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <unordered_map>
#include <mutex>
#include <cstdint>

static void throwRuntime(JNIEnv* env, const std::string& msg) {
  jclass exCls = env->FindClass("java/lang/RuntimeException");
  if (exCls) env->ThrowNew(exCls, msg.c_str());
}

static void checkNvrtc(nvrtcResult r, const char* where) {
  if (r != NVRTC_SUCCESS) {
    std::ostringstream oss;
    oss << where << ": " << nvrtcGetErrorString(r);
    throw std::runtime_error(oss.str());
  }
}

static void checkCu(CUresult r, const char* where) {
  if (r != CUDA_SUCCESS) {
    const char* name = nullptr;
    const char* desc = nullptr;
    cuGetErrorName(r, &name);
    cuGetErrorString(r, &desc);
    std::ostringstream oss;
    oss << where << ": " << (name ? name : "CUDA_ERROR")
        << " - " << (desc ? desc : "unknown");
    throw std::runtime_error(oss.str());
  }
}

static std::string jstringToUtf8(JNIEnv* env, jstring js) {
  if (!js) return {};
  const char* chars = env->GetStringUTFChars(js, nullptr);
  std::string s(chars ? chars : "");
  if (chars) env->ReleaseStringUTFChars(js, chars);
  return s;
}

static std::string compileToPtx(const std::string& src, const std::vector<const char*>& opts) {
  nvrtcProgram prog;
  checkNvrtc(nvrtcCreateProgram(&prog, src.c_str(), "kernel.cu", 0, nullptr, nullptr),
             "nvrtcCreateProgram");

  nvrtcResult cr = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());

  // compile log (useful for debugging kernel strings)
  size_t logSize = 0;
  checkNvrtc(nvrtcGetProgramLogSize(prog, &logSize), "nvrtcGetProgramLogSize");
  std::string log(logSize, '\0');
  if (logSize > 1) checkNvrtc(nvrtcGetProgramLog(prog, log.data()), "nvrtcGetProgramLog");

  if (cr != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&prog);
    std::ostringstream oss;
    oss << "NVRTC compile failed:\n" << log;
    throw std::runtime_error(oss.str());
  }

  size_t ptxSize = 0;
  checkNvrtc(nvrtcGetPTXSize(prog, &ptxSize), "nvrtcGetPTXSize");
  std::string ptx(ptxSize, '\0');
  checkNvrtc(nvrtcGetPTX(prog, ptx.data()), "nvrtcGetPTX");

  nvrtcDestroyProgram(&prog);
  return ptx;
}

// -------- Handle-managed GPU state --------

struct GpuHandle {
  CUdevice dev = 0;
  CUcontext ctx = nullptr;      // primary context retained
  CUmodule mod = nullptr;
  CUfunction fun = nullptr;
};

static std::mutex g_mutex;
static std::unordered_map<jlong, GpuHandle> g_handles;
static std::atomic<long long> g_nextId{1};

static CUcontext g_primaryCtx = nullptr;
static CUdevice g_device = 0;

static void ensureInitialized() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    checkCu(cuInit(0), "cuInit");
    checkCu(cuDeviceGet(&g_device, 0), "cuDeviceGet");
    checkCu(cuDevicePrimaryCtxRetain(&g_primaryCtx, g_device), "cuDevicePrimaryCtxRetain");
    checkCu(cuCtxSetCurrent(g_primaryCtx), "cuCtxSetCurrent");
  });
  
  CUcontext current = nullptr;
  cuCtxGetCurrent(&current);
  if (current != g_primaryCtx) {
    checkCu(cuCtxSetCurrent(g_primaryCtx), "cuCtxSetCurrent");
  }
}

static GpuHandle& getHandleOrThrow(jlong id) {
  auto it = g_handles.find(id);
  if (it == g_handles.end()) {
    throw std::runtime_error("Invalid handle (did you call close already?)");
  }
  return it->second;
}

static void setCurrent(const GpuHandle& h) {
  // Ensure the retained primary context is current on this thread
  CUcontext current = nullptr;
  cuCtxGetCurrent(&current);
  if (current != h.ctx) {
    checkCu(cuCtxSetCurrent(h.ctx), "cuCtxSetCurrent");
  }
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_cuda4s_CudaJitJNI_init(JNIEnv* env, jclass,
                                  jstring kernelSrcJ,
                                  jstring kernelNameJ) {
  try {
    ensureInitialized();
    if (!kernelSrcJ || !kernelNameJ) throw std::runtime_error("kernelSrc/kernelName must not be null");

    const std::string src = jstringToUtf8(env, kernelSrcJ);
    const std::string kname = jstringToUtf8(env, kernelNameJ);

    // Compile once
    std::vector<const char*> nvrtcOpts = { "--std=c++20" };
    std::string ptx = compileToPtx(src, nvrtcOpts);

    // Load module + function once
    CUmodule mod;
    checkCu(cuModuleLoadDataEx(&mod, ptx.c_str(), 0, nullptr, nullptr), "cuModuleLoadDataEx");
    CUfunction fun;
    checkCu(cuModuleGetFunction(&fun, mod, kname.c_str()), "cuModuleGetFunction");

    GpuHandle h;
    h.dev = g_device;
    h.ctx = g_primaryCtx;
    h.mod = mod;
    h.fun = fun;

    // Store handle
    jlong id = (jlong)g_nextId.fetch_add(1);
    {
      std::lock_guard<std::mutex> lk(g_mutex);
      g_handles.emplace(id, std::move(h));
    }
    return id;
  } catch (const std::exception& e) {
    throwRuntime(env, e.what());
    return 0;
  }
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_cuda4s_CudaJitJNI_alloc(JNIEnv* env, jclass, jlong bytes) {
  try {
    ensureInitialized();
    CUdeviceptr dptr;
    checkCu(cuMemAlloc(&dptr, (size_t)bytes), "cuMemAlloc");
    return (jlong)dptr;
  } catch (const std::exception& e) {
    throwRuntime(env, e.what());
    return 0;
  }
}

extern "C"
JNIEXPORT jlongArray JNICALL
Java_com_cuda4s_CudaJitJNI_allocMany(JNIEnv* env, jclass, jlongArray bytesArr) {
  try {
    ensureInitialized();
    jsize n = env->GetArrayLength(bytesArr);
    jlong* bytes = env->GetLongArrayElements(bytesArr, nullptr);
    
    jlongArray resArr = env->NewLongArray(n);
    std::vector<jlong> ptrs(n);
    
    for (jsize i = 0; i < n; ++i) {
      CUdeviceptr dptr;
      checkCu(cuMemAlloc(&dptr, (size_t)bytes[i]), "cuMemAlloc (allocMany)");
      ptrs[i] = (jlong)dptr;
    }
    
    env->SetLongArrayRegion(resArr, 0, n, ptrs.data());
    env->ReleaseLongArrayElements(bytesArr, bytes, JNI_ABORT);
    return resArr;
  } catch (const std::exception& e) {
    throwRuntime(env, e.what());
    return nullptr;
  }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_cuda4s_CudaJitJNI_free(JNIEnv* env, jclass, jlong dptr) {
  try {
    ensureInitialized();
    if (dptr) checkCu(cuMemFree((CUdeviceptr)dptr), "cuMemFree");
  } catch (const std::exception& e) {
    throwRuntime(env, e.what());
  }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_cuda4s_CudaJitJNI_freeMany(JNIEnv* env, jclass, jlongArray ptrsArr) {
  try {
    ensureInitialized();
    jsize n = env->GetArrayLength(ptrsArr);
    jlong* ptrs = env->GetLongArrayElements(ptrsArr, nullptr);
    
    for (jsize i = 0; i < n; ++i) {
      if (ptrs[i]) cuMemFree((CUdeviceptr)ptrs[i]);
    }
    
    env->ReleaseLongArrayElements(ptrsArr, ptrs, JNI_ABORT);
  } catch (const std::exception& e) {
    throwRuntime(env, e.what());
  }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_cuda4s_CudaJitJNI_copyHtoD(JNIEnv* env, jclass, jlong dptr, jfloatArray srcArr, jlong bytes) {
  try {
    ensureInitialized();
    float* hPtr = (float*)env->GetPrimitiveArrayCritical(srcArr, nullptr);
    if (!hPtr) throw std::runtime_error("GetPrimitiveArrayCritical failed");
    CUresult res = cuMemcpyHtoD((CUdeviceptr)dptr, hPtr, (size_t)bytes);
    env->ReleasePrimitiveArrayCritical(srcArr, hPtr, JNI_ABORT);
    checkCu(res, "cuMemcpyHtoD");
  } catch (const std::exception& e) {
    throwRuntime(env, e.what());
  }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_cuda4s_CudaJitJNI_copyDtoH(JNIEnv* env, jclass, jfloatArray destArr, jlong dptr, jlong bytes) {
  try {
    ensureInitialized();
    float* hPtr = (float*)env->GetPrimitiveArrayCritical(destArr, nullptr);
    if (!hPtr) throw std::runtime_error("GetPrimitiveArrayCritical failed");
    CUresult res = cuMemcpyDtoH(hPtr, (CUdeviceptr)dptr, (size_t)bytes);
    env->ReleasePrimitiveArrayCritical(destArr, hPtr, 0);
    checkCu(res, "cuMemcpyDtoH");
  } catch (const std::exception& e) {
    throwRuntime(env, e.what());
  }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_cuda4s_CudaJitJNI_copyHtoDInts(JNIEnv* env, jclass, jlong dptr, jintArray srcArr, jlong bytes) {
  try {
    ensureInitialized();
    jint* hPtr = env->GetIntArrayElements(srcArr, nullptr);
    if (!hPtr) throw std::runtime_error("GetIntArrayElements failed");
    CUresult res = cuMemcpyHtoD((CUdeviceptr)dptr, hPtr, (size_t)bytes);
    env->ReleaseIntArrayElements(srcArr, hPtr, JNI_ABORT);
    checkCu(res, "cuMemcpyHtoD (int)");
  } catch (const std::exception& e) {
    throwRuntime(env, e.what());
  }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_cuda4s_CudaJitJNI_copyDtoHInts(JNIEnv* env, jclass, jintArray destArr, jlong dptr, jlong bytes) {
  try {
    ensureInitialized();
    jint* hPtr = env->GetIntArrayElements(destArr, nullptr);
    if (!hPtr) throw std::runtime_error("GetIntArrayElements failed");
    CUresult res = cuMemcpyDtoH(hPtr, (CUdeviceptr)dptr, (size_t)bytes);
    env->ReleaseIntArrayElements(destArr, hPtr, 0);
    checkCu(res, "cuMemcpyDtoH (int)");
  } catch (const std::exception& e) {
    throwRuntime(env, e.what());
  }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_cuda4s_CudaJitJNI_copyHtoDDoubles(JNIEnv* env, jclass, jlong dptr, jdoubleArray srcArr, jlong bytes) {
  try {
    ensureInitialized();
    jdouble* hPtr = env->GetDoubleArrayElements(srcArr, nullptr);
    if (!hPtr) throw std::runtime_error("GetDoubleArrayElements failed");
    CUresult res = cuMemcpyHtoD((CUdeviceptr)dptr, hPtr, (size_t)bytes);
    env->ReleaseDoubleArrayElements(srcArr, hPtr, JNI_ABORT);
    checkCu(res, "cuMemcpyHtoD (double)");
  } catch (const std::exception& e) {
    throwRuntime(env, e.what());
  }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_cuda4s_CudaJitJNI_copyDtoHDoubles(JNIEnv* env, jclass, jdoubleArray destArr, jlong dptr, jlong bytes) {
  try {
    ensureInitialized();
    jdouble* hPtr = env->GetDoubleArrayElements(destArr, nullptr);
    if (!hPtr) throw std::runtime_error("GetDoubleArrayElements failed");
    CUresult res = cuMemcpyDtoH(hPtr, (CUdeviceptr)dptr, (size_t)bytes);
    env->ReleaseDoubleArrayElements(destArr, hPtr, 0);
    checkCu(res, "cuMemcpyDtoH (double)");
  } catch (const std::exception& e) {
    throwRuntime(env, e.what());
  }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_cuda4s_CudaJitJNI_copyHtoDLongs(JNIEnv* env, jclass, jlong dptr, jlongArray srcArr, jlong bytes) {
  try {
    ensureInitialized();
    jlong* hPtr = env->GetLongArrayElements(srcArr, nullptr);
    if (!hPtr) throw std::runtime_error("GetLongArrayElements failed");
    CUresult res = cuMemcpyHtoD((CUdeviceptr)dptr, hPtr, (size_t)bytes);
    env->ReleaseLongArrayElements(srcArr, hPtr, JNI_ABORT);
    checkCu(res, "cuMemcpyHtoD (long)");
  } catch (const std::exception& e) {
    throwRuntime(env, e.what());
  }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_cuda4s_CudaJitJNI_copyDtoHLongs(JNIEnv* env, jclass, jlongArray destArr, jlong dptr, jlong bytes) {
  try {
    ensureInitialized();
    jlong* hPtr = env->GetLongArrayElements(destArr, nullptr);
    if (!hPtr) throw std::runtime_error("GetLongArrayElements failed");
    CUresult res = cuMemcpyDtoH(hPtr, (CUdeviceptr)dptr, (size_t)bytes);
    env->ReleaseLongArrayElements(destArr, hPtr, 0);
    checkCu(res, "cuMemcpyDtoH (long)");
  } catch (const std::exception& e) {
    throwRuntime(env, e.what());
  }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_cuda4s_CudaJitJNI_launch(JNIEnv* env, jclass, jlong handleId, jint gridX, jint blockX, jlongArray argsArr) {
  try {
    GpuHandle hCopy;
    {
      std::lock_guard<std::mutex> lk(g_mutex);
      hCopy = getHandleOrThrow(handleId);
    }
    setCurrent(hCopy);

    jsize numArgs = env->GetArrayLength(argsArr);
    jlong* argsRaw = env->GetLongArrayElements(argsArr, nullptr);
    
    // We need to pass pointers to the values.
    // For simplicity, we assume all args are either CUdeviceptr or int (mapped to long here)
    // Actually, our VecAdd kernel expects (float*, float*, float*, int).
    // This 'launch' is a bit tricky because cuLaunchKernel expects void** to args.
    
    std::vector<void*> argPointers(numArgs);
    for (int i = 0; i < numArgs; ++i) {
       argPointers[i] = &argsRaw[i];
    }

    checkCu(
      cuLaunchKernel(hCopy.fun,
                     (unsigned int)gridX, 1, 1,
                     (unsigned int)blockX, 1, 1,
                     0, 0,
                     argPointers.data(), nullptr),
      "cuLaunchKernel"
    );
    checkCu(cuCtxSynchronize(), "cuCtxSynchronize");

    env->ReleaseLongArrayElements(argsArr, argsRaw, JNI_ABORT);
  } catch (const std::exception& e) {
    throwRuntime(env, e.what());
  }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_cuda4s_CudaJitJNI_close(JNIEnv* env, jclass, jlong handleId) {
  try {
    GpuHandle h;
    {
      std::lock_guard<std::mutex> lk(g_mutex);
      auto it = g_handles.find(handleId);
      if (it == g_handles.end()) return; // idempotent
      h = it->second;
      g_handles.erase(it);
    }

    setCurrent(h);

    if (h.mod) cuModuleUnload(h.mod);
  } catch (const std::exception& e) {
    throwRuntime(env, e.what());
  }
}
