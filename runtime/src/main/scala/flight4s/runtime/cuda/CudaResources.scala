package flight4s.runtime.cuda

import java.nio.{ByteBuffer, ByteOrder}

import flight4s.core.abi.{DeviceAddress, NativeLaunchRequest}
import flight4s.core.codegen.GeneratedKernel
import flight4s.core.compiler.{ComputeCapability, NvrtcArtifact}
import flight4s.core.ir.{DeviceBuffer, KernelInvocation, KernelSignature}
import flight4s.core.launch.LaunchConfig
import flight4s.core.types.CudaType
import flight4s.runtime.cuda.internal.*

final case class CudaDriverFailure(
    operation: String,
    resultCode: Int,
    resultName: String,
    resultDescription: String,
    infoLog: String,
    errorLog: String
):
  def message: String =
    s"$operation failed with $resultName ($resultCode): $resultDescription"

final class CudaDriverException(
    val failure: CudaDriverFailure
) extends RuntimeException(failure.message)

sealed trait CudaLaunchFailure:
  def message: String

object CudaLaunchFailure:
  final case class InvocationMismatch(
      expectedKernel: String,
      actualKernel: String
  ) extends CudaLaunchFailure:
    override val message: String =
      s"kernel invocation $actualKernel does not belong to $expectedKernel"

  final case class DynamicSharedMemoryRequired(
      kernelName: String,
      elementSizeBytes: Int
  ) extends CudaLaunchFailure:
    override val message: String =
      s"kernel $kernelName requires dynamic shared memory for " +
        s"$elementSizeBytes-byte elements"

  final case class InvalidDynamicSharedMemorySize(
      kernelName: String,
      configuredBytes: Int,
      elementSizeBytes: Int,
      elementAlignmentBytes: Int
  ) extends CudaLaunchFailure:
    override val message: String =
      s"kernel $kernelName dynamic shared memory size " +
        s"$configuredBytes is incompatible with $elementSizeBytes-byte " +
        s"elements aligned to $elementAlignmentBytes bytes"

  final case class Driver(
      failure: CudaDriverFailure
  ) extends CudaLaunchFailure:
    override def message: String = failure.message

final class CudaContext private (
    val deviceOrdinal: Int,
    val computeCapability: ComputeCapability,
    private val handle: Long,
    private val backend: CudaDriverBackend
) extends AutoCloseable:
  private val lifecycleLock = new Object
  private var closed = false
  private var resources = Vector.empty[AutoCloseable]

  def isOpen: Boolean =
    lifecycleLock.synchronized(!closed)

  def load(
      artifact: NvrtcArtifact
  ): Either[CudaDriverFailure, CudaModule] =
    lifecycleLock.synchronized {
      requireOpen()
      val result = backend.loadPtx(handle, artifact.ptx)
      if result.status.succeeded then
        require(
          result.handle != 0L,
          "successful CUDA module load returned a null handle"
        )
        val module =
          new CudaModule(this, artifact, result.handle, backend)
        resources :+= module
        Right(module)
      else
        Left(
          CudaDriverFailure.fromStatus(
            "CUDA PTX module load",
            result.status
          )
        )
    }

  def allocate[T](
      elementCount: Int
  )(using valueType: CudaType[T])
      : Either[CudaDriverFailure, CudaDeviceBuffer[T]] =
    lifecycleLock.synchronized {
      requireOpen()
      require(
        elementCount > 0,
        s"CUDA device buffer element count must be positive: $elementCount"
      )
      val sizeBytes =
        Math.multiplyExact(
          elementCount.toLong,
          valueType.sizeBytes.toLong
        )
      val result = backend.allocateDeviceMemory(handle, sizeBytes)
      if result.status.succeeded then
        require(
          result.handle != 0L,
          "successful CUDA device allocation returned a null address"
        )
        val buffer = new CudaDeviceBuffer(
          context = this,
          valueType = valueType,
          elementCount = elementCount,
          sizeBytes = sizeBytes,
          handle = result.handle,
          backend = backend
        )
        resources :+= buffer
        Right(buffer)
      else
        Left(
          CudaDriverFailure.fromStatus(
            s"CUDA device allocation of $sizeBytes bytes",
            result.status
          )
        )
    }

  override def close(): Unit =
    lifecycleLock.synchronized {
      if !closed then
        resources.reverse.foreach(_.close())
        val status = backend.releasePrimaryContext(deviceOrdinal)
        if !status.succeeded then
          throw CudaDriverException(
            CudaDriverFailure.fromStatus(
              "CUDA primary context release",
              status
            )
          )
        closed = true
    }

  private[cuda] def synchronizedLifecycle[A](body: => A): A =
    lifecycleLock.synchronized(body)

  private[cuda] def requireOpen(): Unit =
    if closed then
      throw IllegalStateException("CUDA context is closed")

  private[cuda] def unregister(resource: AutoCloseable): Unit =
    resources = resources.filterNot(_ eq resource)

  private[flight4s] def nativeHandle: Long =
    lifecycleLock.synchronized {
      requireOpen()
      handle
    }

object CudaContext:
  def open(
      deviceOrdinal: Int
  ): Either[CudaDriverFailure, CudaContext] =
    open(deviceOrdinal, NativeCudaDriver)

  private[cuda] def open(
      deviceOrdinal: Int,
      backend: CudaDriverBackend
  ): Either[CudaDriverFailure, CudaContext] =
    require(
      deviceOrdinal >= 0,
      "CUDA device ordinal must not be negative"
    )
    val result = backend.retainPrimaryContext(deviceOrdinal)
    if result.status.succeeded then
      require(
        result.handle != 0L,
        "successful CUDA context retain returned a null handle"
      )
      Right(
        new CudaContext(
          deviceOrdinal = result.deviceOrdinal,
          computeCapability = ComputeCapability(
            result.computeCapabilityMajor,
            result.computeCapabilityMinor
          ),
          handle = result.handle,
          backend = backend
        )
      )
    else
      Left(
        CudaDriverFailure.fromStatus(
          "CUDA primary context retain",
          result.status
        )
      )

final class CudaDeviceBuffer[T] private[cuda] (
    val context: CudaContext,
    val valueType: CudaType[T],
    val elementCount: Int,
    val sizeBytes: Long,
    private val handle: Long,
    private val backend: CudaDriverBackend
) extends DeviceBuffer[T],
      AutoCloseable:
  private var closed = false

  def isOpen: Boolean =
    context.synchronizedLifecycle(!closed && context.isOpen)

  def copyFrom(
      values: Array[T]
  )(using codec: CudaHostCodec[T]): Either[CudaDriverFailure, Unit] =
    requireHostCodec(codec)
    require(
      values.length == elementCount,
      s"host element count ${values.length} does not match " +
        s"device buffer element count $elementCount"
    )
    requireWholeBufferHostCopy()
    val source = codec.encode(values)
    context.synchronizedLifecycle {
      requireOpen()
      copyResult(
        "CUDA host-to-device copy",
        backend.copyHostToDevice(
          context.nativeHandle,
          handle,
          source
        )
      )
    }

  def copyToArray()(using
      codec: CudaHostCodec[T]
  ): Either[CudaDriverFailure, Array[T]] =
    requireHostCodec(codec)
    requireWholeBufferHostCopy()
    val destination = ByteBuffer
      .allocateDirect(sizeBytes.toInt)
      .order(ByteOrder.nativeOrder())
    context.synchronizedLifecycle {
      requireOpen()
      val status = backend.copyDeviceToHost(
        context.nativeHandle,
        handle,
        destination
      )
      if status.succeeded then
        Right(codec.decode(destination, elementCount))
      else
        Left(
          CudaDriverFailure.fromStatus(
            "CUDA device-to-host copy",
            status
          )
        )
    }

  override def close(): Unit =
    context.synchronizedLifecycle {
      if !closed then
        context.requireOpen()
        val status =
          backend.freeDeviceMemory(context.nativeHandle, handle)
        if !status.succeeded then
          throw CudaDriverException(
            CudaDriverFailure.fromStatus(
              "CUDA device memory free",
              status
            )
          )
        closed = true
        context.unregister(this)
    }

  override private[flight4s] def deviceAddress: DeviceAddress =
    context.synchronizedLifecycle {
      requireOpen()
      DeviceAddress.fromRaw(handle)
    }

  private def requireOpen(): Unit =
    context.requireOpen()
    if closed then
      throw IllegalStateException("CUDA device buffer is closed")

  private def requireHostCodec(codec: CudaHostCodec[T]): Unit =
    require(
      codec.cudaType eq valueType,
      s"host codec ${codec.cudaType.cudaName} does not match " +
        s"device buffer type ${valueType.cudaName}"
    )

  private def requireWholeBufferHostCopy(): Unit =
    require(
      sizeBytes <= Int.MaxValue,
      s"whole-buffer host copies require at most ${Int.MaxValue} bytes: " +
        sizeBytes
    )

  private def copyResult(
      operation: String,
      status: NativeCudaDriverStatus
  ): Either[CudaDriverFailure, Unit] =
    if status.succeeded then Right(())
    else Left(CudaDriverFailure.fromStatus(operation, status))

final class CudaModule private[cuda] (
    val context: CudaContext,
    val artifact: NvrtcArtifact,
    private val handle: Long,
    private val backend: CudaDriverBackend
) extends AutoCloseable:
  private var closed = false

  def isOpen: Boolean =
    context.synchronizedLifecycle(!closed && context.isOpen)

  def function[Args <: Tuple](
      generated: GeneratedKernel[Args]
  ): Either[CudaDriverFailure, CudaFunction[Args]] =
    context.synchronizedLifecycle {
      requireOpen()
      require(
        owns(generated),
        s"kernel ${generated.name} does not belong to this CUDA module artifact"
      )
      val result =
        backend.resolveFunction(context.nativeHandle, handle, generated.name)
      if result.status.succeeded then
        require(
          result.handle != 0L,
          "successful CUDA function lookup returned a null handle"
        )
        Right(new CudaFunction(this, generated, result.handle))
      else
        Left(
          CudaDriverFailure.fromStatus(
            s"CUDA function lookup for ${generated.name}",
            result.status
          )
        )
    }

  override def close(): Unit =
    context.synchronizedLifecycle {
      if !closed then
        context.requireOpen()
        val status =
          backend.unloadModule(context.nativeHandle, handle)
        if !status.succeeded then
          throw CudaDriverException(
            CudaDriverFailure.fromStatus(
              "CUDA module unload",
              status
            )
          )
        closed = true
        context.unregister(this)
    }

  private[cuda] def requireOpen(): Unit =
    context.requireOpen()
    if closed then
      throw IllegalStateException("CUDA module is closed")

  private[cuda] def submit(
      functionHandle: Long,
      request: NativeLaunchRequest
  ): NativeCudaDriverStatus =
    backend.launchKernel(
      contextHandle = context.nativeHandle,
      functionHandle = functionHandle,
      streamHandle = 0L,
      request = request
    )

  private def owns[Args <: Tuple](
      generated: GeneratedKernel[Args]
  ): Boolean =
    artifact.generated.kernels.exists { candidate =>
      candidate.name == generated.name &&
      (candidate.signature eq generated.signature)
    }

final class CudaFunction[Args <: Tuple] private[cuda] (
    val module: CudaModule,
    val generated: GeneratedKernel[Args],
    private val handle: Long
):
  def name: String = generated.name
  def signature: KernelSignature[Args] = generated.signature
  def isValid: Boolean = module.isOpen

  def launch(
      invocation: KernelInvocation[Args],
      config: LaunchConfig
  ): Either[CudaLaunchFailure, Unit] =
    module.context.synchronizedLifecycle {
      module.requireOpen()
      validateInvocation(invocation).flatMap { _ =>
        validateDynamicSharedMemory(config).flatMap { _ =>
          val request = invocation.nativeLaunchRequest(config)
          val status = module.submit(handle, request)
          if status.succeeded then Right(())
          else
            Left(
              CudaLaunchFailure.Driver(
                CudaDriverFailure.fromStatus(
                  s"CUDA kernel launch for $name",
                  status
                )
              )
            )
        }
      }
    }

  private[flight4s] def nativeHandle: Long =
    module.context.synchronizedLifecycle {
      module.requireOpen()
      handle
    }

  private def validateInvocation(
      invocation: KernelInvocation[Args]
  ): Either[CudaLaunchFailure, Unit] =
    if invocation.kernel.name == name &&
        (invocation.kernel.signature eq signature)
    then Right(())
    else
      Left(
        CudaLaunchFailure.InvocationMismatch(
          expectedKernel = name,
          actualKernel = invocation.kernel.name
        )
      )

  private def validateDynamicSharedMemory(
      config: LaunchConfig
  ): Either[CudaLaunchFailure, Unit] =
    generated.launchRequirements.dynamicSharedMemory match
      case None => Right(())
      case Some(requirement)
          if config.dynamicSharedMemoryBytes == 0 =>
        Left(
          CudaLaunchFailure.DynamicSharedMemoryRequired(
            kernelName = name,
            elementSizeBytes = requirement.elementSizeBytes
          )
        )
      case Some(requirement)
          if config.dynamicSharedMemoryBytes <
              requirement.elementSizeBytes ||
            config.dynamicSharedMemoryBytes %
                requirement.elementSizeBytes != 0 ||
            config.dynamicSharedMemoryBytes %
                requirement.elementAlignmentBytes != 0 =>
        Left(
          CudaLaunchFailure.InvalidDynamicSharedMemorySize(
            kernelName = name,
            configuredBytes = config.dynamicSharedMemoryBytes,
            elementSizeBytes = requirement.elementSizeBytes,
            elementAlignmentBytes = requirement.elementAlignmentBytes
          )
        )
      case Some(_) => Right(())

object CudaDriverFailure:
  private[cuda] def fromStatus(
      operation: String,
      status: NativeCudaDriverStatus
  ): CudaDriverFailure =
    CudaDriverFailure(
      operation = operation,
      resultCode = status.resultCode,
      resultName = status.resultName,
      resultDescription = status.resultDescription,
      infoLog = status.infoLog,
      errorLog = status.errorLog
    )
