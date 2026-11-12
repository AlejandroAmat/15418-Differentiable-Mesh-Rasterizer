#include "optimizer.h"
#include <cmath>

Adam::Adam(float lr, float b1, float b2, float eps)
    : m_lr(lr), m_b1(b1), m_b2(b2), m_eps(eps) {}

void Adam::step(float *params, const float *grad, size_t n) {
  if (m_m.size() != n) { // for subdivision / resize
    m_m.assign(n, 0.0f);
    m_v.assign(n, 0.0f);
    m_t = 0;
  }
  m_t++;
  float c1 = 1.0f - std::pow(m_b1, m_t);
  float c2 = 1.0f - std::pow(m_b2, m_t);

  for (size_t i = 0; i < n; i++) {
    float g = grad[i];
    m_m[i] = m_b1 * m_m[i] + (1.0f - m_b1) * g;
    m_v[i] = m_b2 * m_v[i] + (1.0f - m_b2) * g * g;
    params[i] -= m_lr * (m_m[i] / c1) / (std::sqrt(m_v[i] / c2) + m_eps);
  }
}

void Adam::step(std::vector<Vec3F> &params, const std::vector<Vec3F> &grad) {
  step(params[0].m, grad[0].m, params.size() * 3);
}