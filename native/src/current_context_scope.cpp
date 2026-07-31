#include "flight4s/cuda/current_context_scope.hpp"

namespace flight4s::cuda {

CurrentContextScope::CurrentContextScope(CUcontext context)
    : context_(context),
      push_result_(cuCtxPushCurrent(context)),
      active_(push_result_ == CUDA_SUCCESS) {}

CurrentContextScope::~CurrentContextScope() {
  if (active_) {
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);
  }
}

CUresult CurrentContextScope::push_result() const noexcept {
  return push_result_;
}

CUresult CurrentContextScope::close() noexcept {
  if (!active_) {
    return push_result_;
  }

  CUcontext popped = nullptr;
  const auto result = cuCtxPopCurrent(&popped);
  active_ = false;
  if (result != CUDA_SUCCESS) {
    return result;
  }
  return popped == context_ ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CONTEXT;
}

}  // namespace flight4s::cuda
