# Flight4s CUDA Native Launcher

This directory builds the internal JNI library that validates a typed Flight4s
launch request, constructs CUDA's `void**` kernel parameter table, and calls
`cuLaunchKernelEx`.

The launcher does not synchronize the CUDA context or stream. Context, module,
function, stream, and allocation ownership remain runtime responsibilities.

## Requirements

- CMake 3.24 or newer
- a C++17 compiler
- JDK headers
- CUDA Toolkit 12.0 or newer
- an NVIDIA driver compatible with the selected toolkit

## Build And Contract Tests

```shell
cmake -S native -B native/build -DBUILD_TESTING=ON
cmake --build native/build --config Release
ctest --test-dir native/build -C Release --output-on-failure
```

`flight4s_launch_contract` does not require a GPU. It verifies launch geometry,
stable descriptor codes, argument alignment and bounds, and `void**` pointer
construction.

## GPU Integration Test

The optional integration test compiles a small PTX kernel and executes ordinary
and clustered launches through `cuLaunchKernelEx`. It requires a CUDA device
with compute capability 9.0 or newer.

```shell
cmake -S native -B native/build \
  -DBUILD_TESTING=ON \
  -DFLIGHT4S_BUILD_CUDA_INTEGRATION_TESTS=ON
cmake --build native/build --config Release
ctest --test-dir native/build -C Release --output-on-failure
```

## JNI Smoke Test

Pass the absolute native library path to the runtime test:

```shell
sbt -Dflight4s.cuda.native.path=/absolute/path/to/flight4s_cuda \
  "runtime/testOnly flight4s.runtime.cuda.internal.NativeCudaLauncherSuite"
```

Without this property, portable Scala test runs skip the JNI-dependent tests.
Native library packaging and platform classifiers will be added with the public
runtime resource API.
