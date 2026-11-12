#pragma once
#include "image.h"
#include "utils.h"
#include <string>

// The 4 texels a bilinear lookup touches, and their weights (sum to 1)
struct BilinearSample {
  int x[4], y[4];
  float w[4];
};

class Texture {
public:
  void load(const std::string &path);

  int get_w() const { return m_img.get_w(); }
  int get_h() const { return m_img.get_h(); }
  Image &image() { return m_img; }
  const Image &image() const { return m_img; }
  void init(int w, int h, float val);

  BilinearSample bilinear(float u, float v) const;
  Vec3F sample(float u, float v) const;

private:
  Image m_img{1, 1, 3};
};