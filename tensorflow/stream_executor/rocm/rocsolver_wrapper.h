/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file wraps rocsolver API calls with dso loader so that we don't need to
// have explicit linking to librocsolver. All TF hipsarse API usage should route
// through this wrapper.

#ifndef TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCSOLVER_WRAPPER_H_
#define TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCSOLVER_WRAPPER_H_

#include "rocm/include/rocsolver.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace tensorflow {
namespace wrap {

#ifdef PLATFORM_GOOGLE

#define ROCSOLVER_API_WRAPPER(__name)            \
  struct WrapperShim__##__name {                 \
    template <typename... Args>                  \
    rocblas_status operator()(Args... args) {    \
      rocblas_status retval = ::__name(args...); \
      return retval;                             \
    }                                            \
  } __name;

#else

#define ROCSOLVER_API_WRAPPER(__name)                                          \
  struct DynLoadShim__##__name {                                               \
    static const char* kName;                                                  \
    using FuncPtrT = std::add_pointer<decltype(::__name)>::type;               \
    static void* GetDsoHandle() {                                              \
      auto s =                                                                 \
          stream_executor::internal::CachedDsoLoader::GetRocsolverDsoHandle(); \
      return s.ValueOrDie();                                                   \
    }                                                                          \
    static FuncPtrT LoadOrDie() {                                              \
      void* f;                                                                 \
      auto s =                                                                 \
          Env::Default()->GetSymbolFromLibrary(GetDsoHandle(), kName, &f);     \
      CHECK(s.ok()) << "could not find " << kName                              \
                    << " in rocsolver DSO; dlerror: " << s.error_message();    \
      return reinterpret_cast<FuncPtrT>(f);                                    \
    }                                                                          \
    static FuncPtrT DynLoad() {                                                \
      static FuncPtrT f = LoadOrDie();                                         \
      return f;                                                                \
    }                                                                          \
    template <typename... Args>                                                \
    rocblas_status operator()(Args... args) {                                  \
      return DynLoad()(args...);                                               \
    }                                                                          \
  } __name;                                                                    \
  const char* DynLoadShim__##__name::kName = #__name;

#endif

// clang-format off
#define FOREACH_ROCSOLVER_API(__macro)		\
  __macro(rocsolver_spotrf)			\
  __macro(rocsolver_dpotrf)			\
  __macro(rocsolver_cpotrf)			\
  __macro(rocsolver_zpotrf)

// clang-format on

FOREACH_ROCSOLVER_API(ROCSOLVER_API_WRAPPER)

#undef FOREACH_ROCSOLVER_API
#undef ROCSOLVER_API_WRAPPER

}  // namespace wrap
}  // namespace tensorflow

#endif  // TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCSOLVER_WRAPPER_H_
