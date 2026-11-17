#pragma once
#include <cstddef>
#include <cuda_runtime.h>

template <typename T> struct DeviceBuffer {
  T *ptr = nullptr;
  size_t n = 0;
  size_t cap = 0;

  DeviceBuffer() = default;
  ~DeviceBuffer() { cudaFree(ptr); }

  void resize(size_t count) {
    if (n == count)
      return;
    cudaFree(ptr);
    n = count;
    cudaMalloc(&ptr, n * sizeof(T));
  }

  void reserve(size_t count) {
    n = count;
    if (count <= cap)
      return;
    cap = n * 1.5;
    cudaFree(ptr);
    cudaMalloc(&ptr, cap * sizeof(T));
  }

  void upload_to_device(const T *host, size_t count) {
    resize(count);
    cudaMemcpy(ptr, host, n * sizeof(T), cudaMemcpyHostToDevice);
  }

  void get_from_device(T *host) {
    cudaMemcpy(host, ptr, n * sizeof(T), cudaMemcpyDeviceToHost);
  }

  void copy_from_device(T *host, size_t offset, size_t size) {
    cudaMemcpy(host, ptr + offset, size * sizeof(T), cudaMemcpyDeviceToHost);
  }

  void set_zero() { cudaMemset(ptr, 0, n * sizeof(T)); }
};