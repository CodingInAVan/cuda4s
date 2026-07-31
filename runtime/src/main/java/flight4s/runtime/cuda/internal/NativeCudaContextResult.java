package flight4s.runtime.cuda.internal;

final class NativeCudaContextResult {
  private final long handle;
  private final int deviceOrdinal;
  private final int computeCapabilityMajor;
  private final int computeCapabilityMinor;
  private final int resultCode;
  private final byte[] resultNameUtf8;
  private final byte[] resultDescriptionUtf8;

  NativeCudaContextResult(
      long handle,
      int deviceOrdinal,
      int computeCapabilityMajor,
      int computeCapabilityMinor,
      int resultCode,
      byte[] resultNameUtf8,
      byte[] resultDescriptionUtf8) {
    this.handle = handle;
    this.deviceOrdinal = deviceOrdinal;
    this.computeCapabilityMajor = computeCapabilityMajor;
    this.computeCapabilityMinor = computeCapabilityMinor;
    this.resultCode = resultCode;
    this.resultNameUtf8 = resultNameUtf8;
    this.resultDescriptionUtf8 = resultDescriptionUtf8;
  }

  long handle() {
    return handle;
  }

  int deviceOrdinal() {
    return deviceOrdinal;
  }

  int computeCapabilityMajor() {
    return computeCapabilityMajor;
  }

  int computeCapabilityMinor() {
    return computeCapabilityMinor;
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
}
