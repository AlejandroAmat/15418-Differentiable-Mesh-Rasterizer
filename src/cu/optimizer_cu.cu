#include "optimizer_cu.h"

AdamCUDA::AdamCUDA(float lr, float b1, float b2, float eps)
    : m_lr(lr), m_b1(b1), m_b2(b2), m_eps(eps) {}

// create struct of params TODO
__global__ void _step(const float *grads, size_t n, float *_m_m, float *_m_v,
                      float c1, float c2, float lr, float b2, float b1,
                      float eps, float *params) {

  int id = threadIdx.x + blockDim.x * blockIdx.x;

  if (id >= n)
    return;

  float grad = grads[id];

  float m = b1 * _m_m[id] + (1.0f - b1) * grad;
  float v = b2 * _m_v[id] + (1.0f - b2) * grad * grad;
  _m_m[id] = m;
  _m_v[id] = v;
  params[id] -= lr * (m * c1) / (sqrtf(v * c2) + eps);
}

void AdamCUDA::step(float *params, const float *grads, size_t n) {

  if (m_m.n != n) {
    m_m.resize(n);
    m_m.set_zero();
    m_v.resize(n);
    m_v.set_zero();
    m_t = 0;
  }

  m_t++;
  float c1 = 1.0 / (1.0f - std::pow(m_b1, m_t));
  float c2 = 1.0f / (1.0f - std::pow(m_b2, m_t));

  dim3 block(BLOCK_SIZE);
  dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  _step<<<grid, block>>>(grads, n, m_m.ptr, m_v.ptr, c1, c2, m_lr, m_b2, m_b1,
                         m_eps, params);
}