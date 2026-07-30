extern "C" __global__ void write_values(int* output, int base) {
  const int index =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  output[index] = base + index;
}
