#pragma once
#include "image.h"
#include "mesh.h"
#include "persp_camera.h"
#include <algorithm>
#include <cmath>
#include <vector>
// we have to think of image
struct Accumulator {
  std::vector<float> m;
  int _w;
  void reset(int w, int h) {
    m.resize(w * h);
    _w = w;
    std::fill(m.begin(), m.end(), 1.0f);
  }
  float &at(int i, int j) { return m[i + j * _w]; }
  float at(int i, int j) const { return m[i + j * _w]; }
};
struct Face {
  Vec3I vert_idx;
  bool shown;
  Vec2F e1, e0;
  float d;
  int bbox[4];
};
class SoftRasterizerCPU {
public:
  SoftRasterizerCPU() {}
  void forward(Mesh &m, int view_idx);
  void backward();
  void render_shaded(Mesh &m, int view_idx, Image &out);

  void add_camera(const PerspectiveCamera &cam) {
    if (m_cameras.empty()) {
      m_h = cam.get_h();
      m_w = cam.get_w();
    }
    m_cameras.push_back(cam);
  }

  int get_num_cameras() const { return m_cameras.size(); }

  Image &get_image() { return m_render_target; }
  PerspectiveCamera &get_cam(int idx) { return m_cameras[idx]; }

  std::vector<Vec3F> projected_verts;
  std::vector<Face> faces;

  float epsilon = 1e-4;
  float sigma = 12.0f;
  float inv_sigma = 1.0 / sigma;

  Accumulator accum;

private:
  int m_w;
  int m_h;
  std::vector<PerspectiveCamera> m_cameras;
  Image m_render_target{WIDTH, HEIGHT};
};