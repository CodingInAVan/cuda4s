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

  static native int launchKernel(
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
}
