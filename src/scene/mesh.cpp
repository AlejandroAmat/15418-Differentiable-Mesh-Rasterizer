#include "mesh.h"

Vec3F Mesh::get_center() {
  Vec3F center{0, 0, 0};
  for (const Vec3F &v : m_vertices) {
    center += v;
  }
  return center * (1.0f / (float)m_vertices.size());
}

Vec3F Mesh::get_center_bbox(float &scale) {
  Vec3F lo = m_vertices[0], hi = m_vertices[0];
  for (const Vec3F &v : m_vertices) {
    for (int k = 0; k < 3; k++) {
      lo.at(k) = std::min(lo.at(k), v.at(k));
      hi.at(k) = std::max(hi.at(k), v.at(k));
    }
  }
  scale = std::sqrt((hi - lo).dot(hi - lo));
  return (lo + hi) * 0.5f;
}

void Mesh::subdivide_midpoint() {
  std::map<std::pair<int, int>, int> mid;
  std::vector<Vec3I> new_faces;
  new_faces.reserve(m_faces.size() * 4);

  auto midpoint = [&](int a, int b) {
    std::pair<int, int> key = std::minmax(a, b);
    auto it = mid.find(key);
    if (it != mid.end())
      return it->second;
    Vec3F pa = m_vertices[a];
    Vec3F pb = m_vertices[b];
    m_vertices.push_back((pa + pb) * 0.5f);
    int idx = (int)m_vertices.size() - 1;
    mid[key] = idx;
    return idx;
  };

  for (const Vec3I &f : m_faces) {
    int a = f.at(0);
    int b = f.at(1);
    int c = f.at(2);

    int ab = midpoint(a, b);
    int bc = midpoint(b, c);
    int ca = midpoint(c, a);

    new_faces.push_back({a, ab, ca});
    new_faces.push_back({ab, b, bc});
    new_faces.push_back({ca, bc, c});
    new_faces.push_back({ab, bc, ca});
  }
  m_faces.swap(new_faces);
}