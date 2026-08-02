package flight4s.core.compiler

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.HexFormat

import flight4s.core.codegen.{CudaCodegen, GeneratedCudaModule}

final case class NvrtcCompilationKey private (hex: String):
  require(
    hex.length == 64 && hex.forall { character =>
      character >= '0' && character <= '9' ||
      character >= 'a' && character <= 'f'
    },
    "an NVRTC compilation key must be 64 lowercase hexadecimal characters"
  )

  override def toString: String = hex

object NvrtcCompilationKey:
  private val EncodingVersion = 1

  def derive(
      generated: GeneratedCudaModule,
      target: ComputeCapability,
      nvrtcVersion: NvrtcVersion,
      programName: String,
      codegenVersion: Int = CudaCodegen.ArtifactVersion
  ): NvrtcCompilationKey =
    require(
      generated.cudaSource.nonEmpty,
      "generated CUDA source must not be empty"
    )
    require(programName.nonEmpty, "program name must not be empty")
    require(
      !programName.contains('\u0000'),
      "program name must not contain a null character"
    )
    require(codegenVersion > 0, "codegen version must be positive")

    val digest = MessageDigest.getInstance("SHA-256")
    val encoder = DigestEncoder(digest)
    val compilerOptions = NvrtcCompileOptions.resolve(
      generated.compilerOptions,
      target
    )

    encoder.integer(EncodingVersion)
    encoder.integer(codegenVersion)
    encoder.integer(nvrtcVersion.major)
    encoder.integer(nvrtcVersion.minor)
    encoder.integer(target.major)
    encoder.integer(target.minor)
    encoder.string(programName)
    encoder.string(generated.cudaSource)
    encoder.vector(compilerOptions.values)(encoder.string)
    encoder.vector(generated.kernels) { kernel =>
      encoder.string(kernel.name)
      encoder.vector(kernel.signature.abiDescriptors) { descriptor =>
        encoder.string(descriptor.name)
        encoder.byte(descriptor.abiType.nativeCode)
        encoder.integer(descriptor.abiType.sizeBytes)
        encoder.integer(descriptor.abiType.alignmentBytes)
      }
      encoder.optional(kernel.launchRequirements.dynamicSharedMemory) {
        requirement =>
          encoder.integer(requirement.elementSizeBytes)
          encoder.integer(requirement.elementAlignmentBytes)
      }
    }

    NvrtcCompilationKey(HexFormat.of().formatHex(digest.digest()))

  private final class DigestEncoder(digest: MessageDigest):
    private val integerBuffer =
      ByteBuffer.allocate(java.lang.Integer.BYTES).order(ByteOrder.BIG_ENDIAN)

    def byte(value: Byte): Unit =
      digest.update(value)

    def integer(value: Int): Unit =
      integerBuffer.clear()
      integerBuffer.putInt(value)
      integerBuffer.flip()
      digest.update(integerBuffer)

    def string(value: String): Unit =
      val bytes = value.getBytes(StandardCharsets.UTF_8)
      integer(bytes.length)
      digest.update(bytes)

    def vector[A](values: Vector[A])(write: A => Unit): Unit =
      integer(values.size)
      values.foreach(write)

    def optional[A](value: Option[A])(write: A => Unit): Unit =
      value match
        case Some(present) =>
          byte(1.toByte)
          write(present)
        case None =>
          byte(0.toByte)
