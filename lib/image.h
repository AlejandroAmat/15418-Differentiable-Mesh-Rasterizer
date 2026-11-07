#pragma once
#include "utils.h"
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>
class Image {

public:
  Image(int w, int h);

  int get_w() const { return m_w; }
  int get_h() const { return m_h; }

  float &at(int x, int y) { return m_data[y * m_w + x]; }
  float at(int x, int y) const { return m_data[y * m_w + x]; }
  void write_to_file(const std::string &f_out);
  void read_from_file(const std::string &f_in);

private:
  int m_w;
  int m_h;
  std::vector<float> m_data;
};