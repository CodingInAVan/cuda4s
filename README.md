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
- public `CudaContext`, `CudaModule`, `CudaStream`, `CudaEvent`, and typed
  `CudaFunction[Args]` resources;
- retained CUDA primary contexts with compute-capability discovery;
- typed context-owned `CudaDeviceBuffer[T]` allocation and deterministic
  `AutoCloseable` cleanup;
- typed context-owned `CudaPinnedBuffer[T]` allocation backed by reusable
  page-locked host memory;
- exact whole-buffer and partial-range synchronous transfers between
  same-context pinned and device buffers;
- context-owned completion events with recording, non-blocking queries, host
  synchronization, and same-context stream dependencies;
- exact whole-buffer synchronous host/device copies through native-order direct
  staging storage;
- host codecs for every current scalar type, preserving raw F16, BF16, and FP8
  representations;
- context-owned default-mode and non-blocking CUDA streams with explicit
  synchronization;
- whole-buffer and partial-range asynchronous pinned-memory transfers on
  explicit streams;
- reverse-creation-order cleanup across context-owned modules, buffers, and
  streams;
- typed function resolution that preserves the generated kernel signature;
- typed `CudaFunction.launch` submission from the original
  `KernelInvocation[Args]`;
- source-compatible default-stream launch and same-context explicit-stream
  launch with automatic in-flight resource retention;
- explicit context synchronization through `CudaContext.synchronize()`;
- launch-time kernel provenance and dynamic shared-memory validation;
- context-scoped `cuLaunchKernelEx` with structured CUDA Driver failures;
- structured CUDA Driver failures with PTX JIT information and error logs;
- CUDA declaration emission for constants, global parameters, static and
  dynamic shared memory, and lexical local memory;
- a generated, compiled, and executed typed `vectorAdd` integration test;
- a generated, compiled, and executed dynamic shared-memory block reduction
  with CPU-reference verification;
- golden-source tests, native NVRTC contract tests, JNI integration tests, and
  an optional `nvcc` compilation test.

Typed launches can target CUDA's default stream or an owned explicit stream and
remain asynchronous. `CudaContext.synchronize()` waits for all work in the
context, while `CudaStream.synchronize()` waits for one explicit stream; launch
never synchronizes implicitly. Device-buffer array copies remain
whole-buffer and synchronous through temporary pageable direct staging.
Reusable `CudaPinnedBuffer[T]` storage provides a page-locked synchronous path
without repeated direct-buffer allocation. Its same-context device-buffer
transfers also expose explicit-stream asynchronous overloads backed by
`cuMemcpyHtoDAsync` and `cuMemcpyDtoHAsync`. `CudaEvent` exposes completion
markers through `record`, `query`, and `synchronize`, while
`CudaStream.waitFor` establishes GPU-side stream dependencies. Pinned/device
copies accept independently validated source and destination element ranges.
Asynchronous copies and kernel launches automatically retain participating
pinned buffers, device buffers, and modules until their completion boundary.
Explicit-stream work completes through stream, event, or context completion;
default-stream launches complete through context synchronization. Closing an
in-flight resource makes it unavailable immediately and defers its native
release. Pinned host reads and writes are rejected while a transfer is
outstanding. Closing a stream with tracked work synchronizes before destroying
it, and closing a context synchronizes pending default-stream launches before
native teardown. NVRTC diagnostics retain generated CUDA locations and map them
to the closest known Scala `SourceSpan` while preserving the original compiler
log. Scala 3 call-site capture now populates spans for DSL declarations,
stores, accumulation, structured control flow, reductions, and barriers.
Fine-grained expression/operator spans remain a later increment. Deterministic
NVRTC compilation identity now uses a versioned canonical encoding and SHA-256
over generated CUDA, resolved options, target, compiler/codegen versions,
program name, and kernel ABI/launch metadata. Cache storage and lookup are not
yet implemented. `NvrtcCompiler.version()` queries the loaded compiler version
without creating or compiling a CUDA program, allowing that version to
participate in a cache key before lookup.

## Build

Flight4s requires JDK 17 or newer and sbt:

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
