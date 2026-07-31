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
- explicit compute-capability targets and ordered NVRTC option resolution;
- native NVRTC compilation with inspectable PTX, compiler logs, compiler
  version, target, and exact generated-source provenance;
- structured NVRTC compilation failures that retain source, options, and logs;
- public `CudaContext`, `CudaModule`, `CudaStream`, and typed
  `CudaFunction[Args]` resources;
- retained CUDA primary contexts with compute-capability discovery;
- typed context-owned `CudaDeviceBuffer[T]` allocation and deterministic
  `AutoCloseable` cleanup;
- typed context-owned `CudaPinnedBuffer[T]` allocation backed by reusable
  page-locked host memory;
- exact synchronous transfers between same-context pinned and device buffers;
- exact whole-buffer synchronous host/device copies through native-order direct
  staging storage;
- host codecs for every current scalar type, preserving raw F16, BF16, and FP8
  representations;
- context-owned default-mode and non-blocking CUDA streams with explicit
  synchronization;
- reverse-creation-order cleanup across context-owned modules, buffers, and
  streams;
- typed function resolution that preserves the generated kernel signature;
- typed `CudaFunction.launch` submission from the original
  `KernelInvocation[Args]`;
- source-compatible default-stream launch and same-context explicit-stream
  launch;
- launch-time kernel provenance and dynamic shared-memory validation;
- context-scoped `cuLaunchKernelEx` with structured CUDA Driver failures;
- structured CUDA Driver failures with PTX JIT information and error logs;
- CUDA declaration emission for constants, global parameters, static and
  dynamic shared memory, and lexical local memory;
- a generated, compiled, and executed typed `vectorAdd` integration test;
- golden-source tests, native NVRTC contract tests, JNI integration tests, and
  an optional `nvcc` compilation test.

Typed launches can target CUDA's default stream or an owned explicit stream and
remain asynchronous. `CudaStream.synchronize()` provides an explicit wait;
launch never synchronizes implicitly. Device-buffer array copies remain
whole-buffer and synchronous through temporary pageable direct staging.
Reusable `CudaPinnedBuffer[T]` storage provides a page-locked synchronous path
without repeated direct-buffer allocation. Partial and asynchronous copies,
events, and automatic in-flight resource lifetime tracking are the next runtime
slices. Source-map artifacts retain known IR spans, while automatic Scala
source-position capture and NVRTC diagnostic remapping remain later Scala 3
macro/compiler iterations.

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
