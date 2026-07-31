#pragma once

#include <cuda.h>

namespace flight4s::cuda {

class CurrentContextScope final {
 public:
  explicit CurrentContextScope(CUcontext context);

  CurrentContextScope(const CurrentContextScope&) = delete;
  CurrentContextScope& operator=(const CurrentContextScope&) = delete;
  CurrentContextScope(CurrentContextScope&&) = delete;
  CurrentContextScope& operator=(CurrentContextScope&&) = delete;

  ~CurrentContextScope();

  [[nodiscard]] CUresult push_result() const noexcept;
  [[nodiscard]] CUresult close() noexcept;

 private:
  CUcontext context_;
  CUresult push_result_;
  bool active_;
};

}  // namespace flight4s::cuda
