#include "optimizer.h"
#include <cmath>

Adam::Adam(float lr, float b1, float b2, float eps)
    : m_lr(lr), m_b1(b1), m_b2(b2), m_eps(eps) {}

void Adam::step(std::vector<Vec3F> &params, const std::vector<Vec3F> &grad) {
  if (m_m.size() != params.size()) { // for subdivision
    m_m.assign(params.size(), Vec3F{0, 0, 0});
    m_v.assign(params.size(), Vec3F{0, 0, 0});
    m_t = 0;
  }
  m_t++;
  float c1 = 1.0f - std::pow(m_b1, m_t);
  float c2 = 1.0f - std::pow(m_b2, m_t);

  for (size_t i = 0; i < params.size(); i++) {
    for (int k = 0; k < 3; k++) {
      float g = grad[i].at(k);
      float &m = m_m[i].at(k);
      float &v = m_v[i].at(k);
      m = m_b1 * m + (1.0f - m_b1) * g;
      v = m_b2 * v + (1.0f - m_b2) * g * g;
      params[i].at(k) -= m_lr * (m / c1) / (std::sqrt(v / c2) + m_eps);
    }
  }
}