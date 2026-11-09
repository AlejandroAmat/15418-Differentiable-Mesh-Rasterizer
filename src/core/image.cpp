#include "image.h"

Image::Image(int w, int h) : m_w(w), m_h(h) { m_data.resize(w * h); }

void Image::write_to_file(const std::string &f_out) {
  std::ofstream f(f_out, std::ios::binary);
  f << "P5\n" << m_w << " " << m_h << "\n255\n";
  for (int y = 0; y < m_h; y++) {
    for (int x = 0; x < m_w; x++) {
      f.put((unsigned char)std::lround(std::clamp(at(x, y), 0.0f, 1.0f) *
                                       255.0f));
    }
  }
}

void Image::read_from_file(const std::string &f_in) {
  std::ifstream f(f_in, std::ios::binary);
  std::string magic;
  int maxval;
  f >> magic >> m_w >> m_h >> maxval;
  f.get(); // the '\n'
  m_data.resize((size_t)m_w * m_h);
  for (int y = 0; y < m_h; y++)
    for (int x = 0; x < m_w; x++)
      at(x, y) = (unsigned char)f.get() / 255.0f;
}