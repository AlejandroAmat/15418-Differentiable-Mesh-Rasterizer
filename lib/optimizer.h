#pragma once
#include "utils.h"
#include <vector>

class Adam {
public:
  Adam(float lr, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f);

  void step(std::vector<Vec3F> &params, const std::vector<Vec3F> &grad);
  void set_lr(const float lr) { m_lr = lr; }
  void step(float *params, const float *grad, size_t n);

private:
  float m_lr, m_b1, m_b2, m_eps;
  int m_t = 0;
  std::vector<float> m_m, m_v;
};