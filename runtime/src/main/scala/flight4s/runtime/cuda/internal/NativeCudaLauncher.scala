package flight4s.runtime.cuda.internal

import flight4s.core.abi.NativeLaunchRequest

private[flight4s] object NativeCudaLauncher:
  def launch(
      contextHandle: Long,
      functionHandle: Long,
      streamHandle: Long,
      request: NativeLaunchRequest
  ): NativeCudaDriverResult =
    CudaNativeBindings.launchKernel(
      contextHandle,
      functionHandle,
      streamHandle,
      request.gridX,
      request.gridY,
      request.gridZ,
      request.blockX,
      request.blockY,
      request.blockZ,
      request.dynamicSharedMemoryBytes,
      request.usesCluster,
      request.clusterX,
      request.clusterY,
      request.clusterZ,
      request.argumentBuffer,
      request.argumentOffsets,
      request.argumentDescriptorCodes
    )
