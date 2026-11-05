#pragma once
#include "mesh.h"
#include "persp_camera.h"
#include <vector>
// we have to think of image

class SoftRasterizerCPU {
public:
  SoftRasterizerCPU() {}
  void forward(PerspectiveCamera &cam, Mesh &m);
  void backward();

  void add_camera(const PerspectiveCamera &cam) {
    if (m_cameras.empty()) {
      m_h = cam.get_h();
      m_w = cam.get_w();
    }
    m_cameras.push_back(cam);
  }

  int get_num_cameras() const { return m_cameras.size(); }

private:
  int m_w;
  int m_h;
  std::vector<PerspectiveCamera> m_cameras;
};