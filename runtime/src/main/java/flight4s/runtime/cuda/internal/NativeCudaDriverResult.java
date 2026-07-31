package flight4s.runtime.cuda.internal;

final class NativeCudaDriverResult {
  private final long handle;
  private final int resultCode;
  private final byte[] resultNameUtf8;
  private final byte[] resultDescriptionUtf8;
  private final byte[] infoLogUtf8;
  private final byte[] errorLogUtf8;

  NativeCudaDriverResult(
      long handle,
      int resultCode,
      byte[] resultNameUtf8,
      byte[] resultDescriptionUtf8,
      byte[] infoLogUtf8,
      byte[] errorLogUtf8) {
    this.handle = handle;
    this.resultCode = resultCode;
    this.resultNameUtf8 = resultNameUtf8;
    this.resultDescriptionUtf8 = resultDescriptionUtf8;
    this.infoLogUtf8 = infoLogUtf8;
    this.errorLogUtf8 = errorLogUtf8;
  }

  long handle() {
    return handle;
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
