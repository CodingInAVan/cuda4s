package flight4s.runtime.cuda.internal

import java.nio.charset.StandardCharsets

import flight4s.core.abi.NativeLaunchRequest

private[flight4s] final case class NativeCudaDriverStatus(
    resultCode: Int,
    resultName: String,
    resultDescription: String,
    infoLog: String = "",
    errorLog: String = ""
):
  def succeeded: Boolean = resultCode == 0

private[flight4s] final case class NativeCudaContextRetainResult(
    handle: Long,
    deviceOrdinal: Int,
    computeCapabilityMajor: Int,
    computeCapabilityMinor: Int,
    status: NativeCudaDriverStatus
)

private[flight4s] final case class NativeCudaResourceResult(
    handle: Long,
    status: NativeCudaDriverStatus
)

private[flight4s] trait CudaDriverBackend:
  def retainPrimaryContext(
      deviceOrdinal: Int
  ): NativeCudaContextRetainResult

  def releasePrimaryContext(
      deviceOrdinal: Int
  ): NativeCudaDriverStatus

  def loadPtx(
      contextHandle: Long,
      ptx: IArray[Byte]
  ): NativeCudaResourceResult

  def unloadModule(
      contextHandle: Long,
      moduleHandle: Long
  ): NativeCudaDriverStatus

  def resolveFunction(
      contextHandle: Long,
      moduleHandle: Long,
      functionName: String
  ): NativeCudaResourceResult

  def launchKernel(
      contextHandle: Long,
      functionHandle: Long,
      streamHandle: Long,
      request: NativeLaunchRequest
  ): NativeCudaDriverStatus

private[flight4s] object NativeCudaDriver extends CudaDriverBackend:
  override def retainPrimaryContext(
      deviceOrdinal: Int
  ): NativeCudaContextRetainResult =
    val result =
      CudaNativeBindings.retainPrimaryContext(deviceOrdinal)
    NativeCudaContextRetainResult(
      handle = result.handle(),
      deviceOrdinal = result.deviceOrdinal(),
      computeCapabilityMajor = result.computeCapabilityMajor(),
      computeCapabilityMinor = result.computeCapabilityMinor(),
      status = NativeCudaDriverStatus(
        resultCode = result.resultCode(),
        resultName = decode(result.resultNameUtf8()),
        resultDescription = decode(result.resultDescriptionUtf8())
      )
    )

  override def releasePrimaryContext(
      deviceOrdinal: Int
  ): NativeCudaDriverStatus =
    status(CudaNativeBindings.releasePrimaryContext(deviceOrdinal))

  override def loadPtx(
      contextHandle: Long,
      ptx: IArray[Byte]
  ): NativeCudaResourceResult =
    resource(
      CudaNativeBindings.loadPtx(
        contextHandle,
        ptx.asInstanceOf[Array[Byte]]
      )
    )

  override def unloadModule(
      contextHandle: Long,
      moduleHandle: Long
  ): NativeCudaDriverStatus =
    status(
      CudaNativeBindings.unloadModule(contextHandle, moduleHandle)
    )

  override def resolveFunction(
      contextHandle: Long,
      moduleHandle: Long,
      functionName: String
  ): NativeCudaResourceResult =
    resource(
      CudaNativeBindings.resolveFunction(
        contextHandle,
        moduleHandle,
        functionName.getBytes(StandardCharsets.UTF_8)
      )
    )

  override def launchKernel(
      contextHandle: Long,
      functionHandle: Long,
      streamHandle: Long,
      request: NativeLaunchRequest
  ): NativeCudaDriverStatus =
    status(
      NativeCudaLauncher.launch(
        contextHandle,
        functionHandle,
        streamHandle,
        request
      )
    )

  private def resource(
      result: NativeCudaDriverResult
  ): NativeCudaResourceResult =
    NativeCudaResourceResult(
      handle = result.handle(),
      status = status(result)
    )

  private def status(
      result: NativeCudaDriverResult
  ): NativeCudaDriverStatus =
    NativeCudaDriverStatus(
      resultCode = result.resultCode(),
      resultName = decode(result.resultNameUtf8()),
      resultDescription = decode(result.resultDescriptionUtf8()),
      infoLog = decode(result.infoLogUtf8()),
      errorLog = decode(result.errorLogUtf8())
    )

  private def decode(bytes: Array[Byte]): String =
    String(bytes, StandardCharsets.UTF_8)
