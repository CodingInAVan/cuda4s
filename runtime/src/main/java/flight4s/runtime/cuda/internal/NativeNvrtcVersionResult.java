package flight4s.runtime.cuda.internal;

final class NativeNvrtcVersionResult {
  private final byte[] resultNameUtf8;
  private final int resultCode;
  private final int versionMajor;
  private final int versionMinor;

  NativeNvrtcVersionResult(
      byte[] resultNameUtf8,
      int resultCode,
      int versionMajor,
      int versionMinor) {
    this.resultNameUtf8 = resultNameUtf8;
    this.resultCode = resultCode;
    this.versionMajor = versionMajor;
    this.versionMinor = versionMinor;
  }

  byte[] resultNameUtf8() {
    return resultNameUtf8;
  }

  int resultCode() {
    return resultCode;
  }

  int versionMajor() {
    return versionMajor;
  }

  int versionMinor() {
    return versionMinor;
  }
}
