#pragma once
#include "utils.h"
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

class Image {

public:
  Image(int w, int h, int c = 1);

  int get_w() const { return m_w; }
  int get_h() const { return m_h; }
  int get_c() const { return m_c; }

  float &at(int x, int y, int ch = 0) {
    return m_data[(y * m_w + x) * m_c + ch];
  }
  float at(int x, int y, int ch = 0) const {
    return m_data[(y * m_w + x) * m_c + ch];
  }
  void write_to_file(const std::string &f_out);
  void read_from_file(const std::string &f_in);
  void write_png(const std::string &f_out) const;
  void read_png(const std::string &f_in);
  void fill(float v) { std::fill(m_data.begin(), m_data.end(), v); }

private:
  int m_w;
  int m_h;
  int m_c;
  std::vector<float> m_data;
};