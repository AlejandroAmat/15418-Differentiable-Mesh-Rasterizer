#include "image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

Image::Image(int w, int h, int c) : m_w(w), m_h(h), m_c(c) {
  m_data.resize(w * h * c);
}

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

void Image::write_png(const std::string &f_out) const {
  std::vector<unsigned char> buf((size_t)m_w * m_h * m_c);
  for (size_t i = 0; i < buf.size(); ++i)
    buf[i] =
        (unsigned char)std::lround(std::clamp(m_data[i], 0.0f, 1.0f) * 255.0f);
  stbi_write_png(f_out.c_str(), m_w, m_h, m_c, buf.data(), m_w * m_c);
}

void Image::read_png(const std::string &f_in) {
  int n;
  unsigned char *px = stbi_load(f_in.c_str(), &m_w, &m_h, &n, m_c);
  if (!px)
    throw std::runtime_error("cannot load " + f_in);
  m_data.resize((size_t)m_w * m_h * m_c);
  for (size_t i = 0; i < m_data.size(); ++i)
    m_data[i] = px[i] / 255.0f;
  stbi_image_free(px);
}