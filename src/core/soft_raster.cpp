#include "soft_raster.h"
#include <iostream>
#include <vector>

void SoftRasterizerCPU::forward(PerspectiveCamera &cam, Mesh &m) {

  // Project
  std::vector<Vec3F> projected_verts;

  for (const Vec3F &v : m.vertices()) {
    projected_verts.push_back(cam.T() * v);
  }
}