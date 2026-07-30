package flight4s.runtime.cuda.internal

import java.nio.charset.StandardCharsets

private[flight4s] final case class NativeNvrtcCompilation(
    ptx: Array[Byte],
    compileLog: String,
    resultName: String,
    resultCode: Int,
    versionMajor: Int,
    versionMinor: Int
)

private[flight4s] object NativeNvrtcCompiler:
  def compile(
      cudaSource: String,
      programName: String,
      options: Vector[String]
  ): NativeNvrtcCompilation =
    val nativeResult = CudaNativeBindings.compileCuda(
      cudaSource.getBytes(StandardCharsets.UTF_8),
      programName.getBytes(StandardCharsets.UTF_8),
      options.map(_.getBytes(StandardCharsets.UTF_8)).toArray
    )

    NativeNvrtcCompilation(
      ptx = nativeResult.ptx(),
      compileLog = decode(nativeResult.compileLogUtf8()),
      resultName = decode(nativeResult.resultNameUtf8()),
      resultCode = nativeResult.resultCode(),
      versionMajor = nativeResult.versionMajor(),
      versionMinor = nativeResult.versionMinor()
    )

  private def decode(bytes: Array[Byte]): String =
    String(bytes, StandardCharsets.UTF_8)
