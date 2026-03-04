#!/bin/bash
set -e

# Define project root
PROJECT_ROOT="$(pwd)"
NATIVE_DIR="$PROJECT_ROOT/native"
BUILD_DIR="$NATIVE_DIR/build"

echo "=== Building Native CUDA JNI Library ==="

# 1. Create and enter build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 2. Configure with CMake
echo "Configuring with CMake..."
cmake ..

# 3. Build the project
echo "Compiling..."
make -j$(nproc)

# 4. Copy the library back to the native directory
echo "Copying library to $NATIVE_DIR..."
cp libcudajitjni.so ..

echo "=== Build Successful ==="
echo "To run the project, use:"
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$NATIVE_DIR"
echo "sbt run"
