# Build native JNI library
cd native
make
cd ..

# Make JVM find libvecaddjitjni.so
export LD_LIBRARY_PATH="$PWD/native:$LD_LIBRARY_PATH"

# Also ensure CUDA libs are discoverable if not already (common on some distros)
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

sbt -batch run
