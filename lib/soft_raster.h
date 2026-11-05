#pragma once
#include "persp_camera.h"
#include <vector>

class SoftRasterizerCPU {
public:
  SoftRasterizerCPU(int w, int h) : m_w(w), m_h(h) {}
  void forward();
  void backward();

  void add_camera(const PerspectiveCamera &cam) { m_cameras.push_back(cam); }

private:
  int m_w;
  int m_h;
  std::vector<PerspectiveCamera> m_cameras;
};