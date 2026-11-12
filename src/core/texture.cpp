#include "texture.h"
#include "stb_image.h"
#include <cmath>
#include <stdexcept>

static float srgb_to_linear(float c) {
  return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

void Texture::load(const std::string &path) {
  m_img = Image(1, 1, 3);
  m_img.read_png(path);

  for (int y = 0; y < m_img.get_h(); ++y)
    for (int x = 0; x < m_img.get_w(); ++x)
      for (int c = 0; c < 3; ++c)
        m_img.at(x, y, c) = srgb_to_linear(m_img.at(x, y, c));
}

BilinearSample Texture::bilinear(float u, float v) const {
  int w = get_w(), h = get_h();
  u -= std::floor(u);
  v -= std::floor(v);

  float fx = u * w - 0.5f, fy = v * h - 0.5f;
  int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy);
  float tx = fx - x0, ty = fy - y0;

  auto wrap = [](int i, int n) { return ((i % n) + n) % n; };
  int x1 = wrap(x0 + 1, w), y1 = wrap(y0 + 1, h);
  x0 = wrap(x0, w);
  y0 = wrap(y0, h);

  return {{x0, x1, x0, x1},
          {y0, y0, y1, y1},
          {(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty}};
}

Vec3F Texture::sample(float u, float v) const {
  BilinearSample b = bilinear(u, v);
  float c[3] = {0, 0, 0};
  for (int k = 0; k < 4; ++k)
    for (int ch = 0; ch < 3; ++ch)
      c[ch] += b.w[k] * m_img.at(b.x[k], b.y[k], ch);
  return {c[0], c[1], c[2]};
}

void Texture::init(int w, int h, float val) {
  m_img = Image(w, h, 3);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < 3; ++c)
        m_img.at(x, y, c) = val;
}