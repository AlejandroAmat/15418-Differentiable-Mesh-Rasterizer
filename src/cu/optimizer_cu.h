#pragma once
#include "persp_camera.h"
#include "structs_cu.h"

class AdamCUDA {
public:
  AdamCUDA(float lr, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f);

  void step(float *params, const float *grads, size_t n);
  void set_lr(const float lr) { m_lr = lr; }

private:
  float m_lr, m_b1, m_b2, m_eps;
  int m_t = 0;
  DeviceBuffer<float> m_v;
  DeviceBuffer<float> m_m;
};