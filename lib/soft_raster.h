#pragma once
#include "image.h"
#include "loss.h"
#include "mesh.h"
#include "persp_camera.h"
#include "texture.h"
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
  void forward(Mesh &m, const Texture &tex, int view_idx);
  void backward(LossData &loss, Mesh &mesh, const Texture &tex, int idx,
                std::vector<Vec3F> &grads, Image &tex_grad);

  void render_shaded(Mesh &m, const Texture &tex, int view_idx, Image &out);

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
  std::vector<Vec3F> m_grad_verts;

  void set_sigma(float s) {
    sigma = s;
    inv_sigma = 1.0f / s;
  }

  float epsilon = 1e-4;
  float sigma = 0.1f;
  float inv_sigma = 1.0 / sigma;

  Accumulator accum;

  std::vector<float> m_lmax, m_wsum, m_csum;
  float gamma = 1e-4f;
  float eps_bg = 1e-3f;
  float z_near = 0.1f, z_far = 2.5f;
  Vec3F bg_color{1, 1, 1};

private:
  int m_w;
  int m_h;
  std::vector<PerspectiveCamera> m_cameras;
  Image m_render_target{WIDTH, HEIGHT, 4};
};