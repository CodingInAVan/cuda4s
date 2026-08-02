package flight4s.runtime.cuda.internal;

final class NativeCudaFunctionAttributesResult {
  private final int maxThreadsPerBlock;
  private final int staticSharedMemoryBytes;
  private final int constantMemoryBytes;
  private final int localMemoryBytes;
  private final int registersPerThread;
  private final int resultCode;
  private final byte[] resultNameUtf8;
  private final byte[] resultDescriptionUtf8;
  private final byte[] infoLogUtf8;
  private final byte[] errorLogUtf8;

  NativeCudaFunctionAttributesResult(
      int maxThreadsPerBlock,
      int staticSharedMemoryBytes,
      int constantMemoryBytes,
      int localMemoryBytes,
      int registersPerThread,
      int resultCode,
      byte[] resultNameUtf8,
      byte[] resultDescriptionUtf8,
      byte[] infoLogUtf8,
      byte[] errorLogUtf8) {
    this.maxThreadsPerBlock = maxThreadsPerBlock;
    this.staticSharedMemoryBytes = staticSharedMemoryBytes;
    this.constantMemoryBytes = constantMemoryBytes;
    this.localMemoryBytes = localMemoryBytes;
    this.registersPerThread = registersPerThread;
    this.resultCode = resultCode;
    this.resultNameUtf8 = resultNameUtf8;
    this.resultDescriptionUtf8 = resultDescriptionUtf8;
    this.infoLogUtf8 = infoLogUtf8;
    this.errorLogUtf8 = errorLogUtf8;
  }

  int maxThreadsPerBlock() {
    return maxThreadsPerBlock;
  }

  int staticSharedMemoryBytes() {
    return staticSharedMemoryBytes;
  }

  int constantMemoryBytes() {
    return constantMemoryBytes;
  }

  int localMemoryBytes() {
    return localMemoryBytes;
  }

  int registersPerThread() {
    return registersPerThread;
  }

  int resultCode() {
    return resultCode;
  }

  byte[] resultNameUtf8() {
    return resultNameUtf8;
  }

  byte[] resultDescriptionUtf8() {
    return resultDescriptionUtf8;
  }

  byte[] infoLogUtf8() {
    return infoLogUtf8;
  }

  byte[] errorLogUtf8() {
    return errorLogUtf8;
  }
}
