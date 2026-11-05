#pragma once
#include "utils.h"
#include <vector>

class PerspectiveCamera {

public:
  PerspectiveCamera(int w, int h, const Mat3 &K) : m_w(w), m_h(h), m_K(K) {}

  void set_intrinsics(const Mat3 &K);

  void set_extrinsics(const Mat4 &T);

  int get_w() const { return m_w; }
  int get_h() const { return m_h; }

private:
  int m_w;
  int m_h;
  Mat3 m_K;
  Mat4 m_T;
};