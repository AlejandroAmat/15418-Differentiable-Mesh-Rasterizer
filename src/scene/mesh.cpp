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