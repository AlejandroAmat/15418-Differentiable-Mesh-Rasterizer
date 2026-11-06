#include "image.h"

Image::Image(int w, int h) : m_w(w), m_h(h) { m_data.resize(w * h); }

void Image::write_to_file(const std::string &f_out) {
  std::ofstream f(f_out, std::ios::binary);
  f << "P5\n" << m_w << " " << m_h << "\n255\n";
  for (int y = 0; y < m_h; y++) {
    for (int x = 0; x < m_w; x++) {
      f.put((unsigned char)at(x, y));
    }
  }
}