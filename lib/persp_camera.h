#pragma once
#include "mesh.h"
#include "utils.h"
#include <vector>

class PerspectiveCamera {

public:
  PerspectiveCamera(int w, int h, const Mat3 &K) : m_w(w), m_h(h), m_K(K) {}

  void set_intrinsics(const Mat3 &K);

  void set_extrinsics(const Mat4 &T);

  int get_w() const { return m_w; }
  int get_h() const { return m_h; }
  const Mat4 &T() const { return m_T; };
  Mat4 &T() { return m_T; };
  const Mat3 &K() const { return m_K; };

  void look_at(Vec3F center);
  void update_pos(int i, int n_views, float radius);
  void set_pos(const Vec3F &new_pos) { pos = new_pos; }

private:
  int m_w;
  int m_h;
  Mat3 m_K;
  Mat4 m_T = Mat4::identity();

  Vec3F up_world = {0.0, 1.0, 0.0};
  Vec3F pos = {0, 0, 0};
};