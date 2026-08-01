package flight4s.runtime.cuda.internal;

import java.io.File;
import java.nio.ByteBuffer;

final class CudaNativeBindings {
  private static final String EXPLICIT_LIBRARY_PATH =
      "flight4s.cuda.native.path";

  static {
    String explicitPath = System.getProperty(EXPLICIT_LIBRARY_PATH);
    if (explicitPath == null || explicitPath.isBlank()) {
      System.loadLibrary("flight4s_cuda");
    } else {
      System.load(new File(explicitPath).getAbsolutePath());
    }
  }

  private CudaNativeBindings() {}

  static native NativeCudaDriverResult launchKernel(
      long contextHandle,
      long functionHandle,
      long streamHandle,
      int gridX,
      int gridY,
      int gridZ,
      int blockX,
      int blockY,
      int blockZ,
      int dynamicSharedMemoryBytes,
      boolean usesCluster,
      int clusterX,
      int clusterY,
      int clusterZ,
      ByteBuffer argumentStorage,
      int[] argumentOffsets,
      byte[] argumentDescriptorCodes);

  static native NativeNvrtcResult compileCuda(
      byte[] sourceUtf8,
      byte[] programNameUtf8,
      byte[][] optionsUtf8);

  static native NativeCudaContextResult retainPrimaryContext(
      int deviceOrdinal);

  static native NativeCudaDriverResult releasePrimaryContext(
      int deviceOrdinal);

  static native NativeCudaDriverResult loadPtx(
      long contextHandle,
      byte[] ptx);

  static native NativeCudaDriverResult unloadModule(
      long contextHandle,
      long moduleHandle);

  static native NativeCudaDriverResult resolveFunction(
      long contextHandle,
      long moduleHandle,
      byte[] functionNameUtf8);

  static native NativeCudaDriverResult createStream(
      long contextHandle,
      int flags);

  static native NativeCudaDriverResult destroyStream(
      long contextHandle,
      long streamHandle);

  static native NativeCudaDriverResult synchronizeStream(
      long contextHandle,
      long streamHandle);

  static native NativeCudaDriverResult createEvent(
      long contextHandle,
      int flags);

  static native NativeCudaDriverResult destroyEvent(
      long contextHandle,
      long eventHandle);

  static native NativeCudaDriverResult recordEvent(
      long contextHandle,
      long eventHandle,
      long streamHandle);

  static native NativeCudaEventQueryResult queryEvent(
      long contextHandle,
      long eventHandle);

  static native NativeCudaDriverResult synchronizeEvent(
      long contextHandle,
      long eventHandle);

  static native NativeCudaDriverResult waitForEvent(
      long contextHandle,
      long streamHandle,
      long eventHandle);

  static native NativeCudaPinnedMemoryResult allocatePinnedMemory(
      long contextHandle,
      long sizeBytes);

  static native NativeCudaDriverResult freePinnedMemory(
      long contextHandle,
      long hostAddress);

  static native NativeCudaDriverResult allocateDeviceMemory(
      long contextHandle,
      long sizeBytes);

  static native NativeCudaDriverResult freeDeviceMemory(
      long contextHandle,
      long deviceAddress);

  static native NativeCudaDriverResult copyHostToDevice(
      long contextHandle,
      long deviceAddress,
      ByteBuffer source);

  static native NativeCudaDriverResult copyDeviceToHost(
      long contextHandle,
      long deviceAddress,
      ByteBuffer destination);
}
