package flight4s.runtime.cuda

import flight4s.core.codegen.GeneratedCudaModule
import flight4s.core.compiler.*
import flight4s.runtime.cuda.internal.NativeNvrtcCompiler

object NvrtcCompiler:
  val DefaultProgramName: String = "flight4s_generated.cu"

  def version(): Either[NvrtcVersionQueryFailure, NvrtcVersion] =
    val nativeResult = NativeNvrtcCompiler.version()
    if nativeResult.resultCode == 0 then
      Right(
        NvrtcVersion(
          nativeResult.versionMajor,
          nativeResult.versionMinor
        )
      )
    else
      Left(
        NvrtcVersionQueryFailure(
          nativeResult.resultCode,
          nativeResult.resultName
        )
      )

  def compile(
      generated: GeneratedCudaModule,
      target: ComputeCapability,
      programName: String = DefaultProgramName
  ): Either[NvrtcCompileFailure, NvrtcArtifact] =
    require(
      generated.cudaSource.nonEmpty,
      "generated CUDA source must not be empty"
    )
    require(
      programName.nonEmpty,
      "NVRTC program name must not be empty"
    )
    require(
      !programName.contains('\u0000'),
      "NVRTC program name must not contain a null character"
    )

    val compilerOptions = NvrtcCompileOptions.resolve(
      generated.compilerOptions,
      target
    )
    val nativeResult = NativeNvrtcCompiler.compile(
      generated.cudaSource,
      programName,
      compilerOptions.values
    )
    val version = NvrtcVersion(
      nativeResult.versionMajor,
      nativeResult.versionMinor
    )

    if nativeResult.resultCode == 0 then
      Right(
        NvrtcArtifact(
          generated = generated,
          ptx = IArray.unsafeFromArray(nativeResult.ptx.clone()),
          compileLog = nativeResult.compileLog,
          nvrtcVersion = version,
          target = target,
          compilerOptions = compilerOptions,
          programName = programName
        )
      )
    else
      Left(
        NvrtcCompileFailure(
          generated = generated,
          resultCode = nativeResult.resultCode,
          resultName = nativeResult.resultName,
          compileLog = nativeResult.compileLog,
          nvrtcVersion = version,
          target = target,
          compilerOptions = compilerOptions,
          programName = programName
        )
      )
