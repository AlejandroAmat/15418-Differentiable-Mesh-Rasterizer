#pragma once
#include <cstddef>
#include <cuda_runtime.h>

template <typename T> struct DeviceBuffer {
  T *ptr = nullptr;
  size_t n = 0;

  DeviceBuffer() = default;
  ~DeviceBuffer() { cudaFree(ptr); }

  void resize(size_t count) {
    if (n == count)
      return;
    cudaFree(ptr);
    n = count;
    cudaMalloc(&ptr, n * sizeof(T));
  }

  void upload_to_device(const T *host, size_t count) {
    resize(count);
    cudaMemcpy(ptr, host, n * sizeof(T), cudaMemcpyHostToDevice);
  }

  void get_from_device(T *host) {
    cudaMemcpy(host, ptr, n * sizeof(T), cudaMemcpyDeviceToHost);
  }

  void set_zero() { cudaMemset(ptr, 0, n * sizeof(T)); }
};