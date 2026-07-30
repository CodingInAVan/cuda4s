package flight4s.runtime.cuda.internal;

final class NativeNvrtcResult {
  private final byte[] ptx;
  private final byte[] compileLogUtf8;
  private final byte[] resultNameUtf8;
  private final int resultCode;
  private final int versionMajor;
  private final int versionMinor;

  NativeNvrtcResult(
      byte[] ptx,
      byte[] compileLogUtf8,
      byte[] resultNameUtf8,
      int resultCode,
      int versionMajor,
      int versionMinor) {
    this.ptx = ptx;
    this.compileLogUtf8 = compileLogUtf8;
    this.resultNameUtf8 = resultNameUtf8;
    this.resultCode = resultCode;
    this.versionMajor = versionMajor;
    this.versionMinor = versionMinor;
  }

  byte[] ptx() {
    return ptx;
  }

  byte[] compileLogUtf8() {
    return compileLogUtf8;
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
