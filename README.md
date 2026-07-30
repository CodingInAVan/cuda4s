# Flight4s

Flight4s is a Scala 3 GPU programming platform built under the
[GPUFlight](https://github.com/gpu-flight) organization.

The first implementation is CUDA-first: its typed DSL and IR generate
inspectable CUDA C++ for compilation with NVRTC. The project name deliberately
does not bind the overall platform to one GPU vendor, leaving room for future
HIP or Metal backends after the CUDA architecture and backend boundary are
proven.

## Status

Flight4s is pre-alpha and under active design. The current implementation
provides:

- CUDA scalar type witnesses, including F16, BF16, and FP8 formats;
- typed expressions, places, statements, control flow, and reductions;
- distinct module constants, rank-aware kernel shared arrays, and lexical local
  arrays;
- module and kernel validation for memory ownership, scope, and access;
- typed kernel signatures and compile-time-checked launch argument tuples;
- ordered CUDA ABI descriptors and exact scalar/device-pointer byte encoding;
- aligned direct launch storage with stable descriptor codes and slot offsets;
- validated 1D, 2D, and 3D grid/block configuration with optional clusters;
- a JNI-facing launch request with reference metadata validation;
- a C++ JNI launcher with native ABI validation and `void**` construction;
- ordinary and clustered `cuLaunchKernelEx` execution without implicit sync;
- structural validation independent of code generation;
- validation-gated deterministic CUDA C++ source generation;
- inspectable generated module and typed kernel artifacts with explicit C++20
  compiler options;
- CUDA declaration emission for constants, global parameters, static and
  dynamic shared memory, and lexical local memory;
- golden-source tests plus an optional `nvcc` compilation test.

NVRTC integration and public CUDA context, module, function, stream, and memory
resource APIs are not implemented yet. Source-map artifacts retain known IR
spans, while automatic Scala source-position capture remains a later Scala 3
macro iteration. The current native launcher is an internal foundation for the
future runtime objects.

## Build

Flight4s requires a JDK and sbt:

```shell
sbt test
```

The optional native CUDA launcher requires CMake, a C++20 compiler (GCC 10+,
Clang 10+, or Visual Studio 2022+), JDK headers, and CUDA Toolkit 12 or newer:

```shell
cmake -S native -B native/build -DBUILD_TESTING=ON
cmake --build native/build --config Release
ctest --test-dir native/build -C Release --output-on-failure
```

See [native/README.md](native/README.md) for JNI and GPU integration tests.

## Coordinates

The planned core artifact is:

```scala
"io.github.gpu-flight" %% "flight4s-core" % "<version>"
```

No release has been published yet.

## License

Flight4s is licensed under the terms in [LICENSE](LICENSE).
