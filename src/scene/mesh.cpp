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

  bool has_uv = !m_uvs.empty() && m_faces_uvs.size() == m_faces.size();
  std::vector<Vec2F> new_uvs;
  std::vector<Vec3I> new_faces_uvs;
  if (has_uv) {
    new_uvs.reserve(m_faces.size() * 6);
    new_faces_uvs.reserve(m_faces.size() * 4);
  }

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

  for (size_t t = 0; t < m_faces.size(); ++t) {

    const Vec3I &f = m_faces[t];
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

    if (has_uv) {
      const Vec3I &fu = m_faces_uvs[t];
      Vec2F ua = m_uvs[fu.at(0)], ub = m_uvs[fu.at(1)], uc = m_uvs[fu.at(2)];
      Vec2F uab = (ua + ub) * 0.5f;
      Vec2F ubc = (ub + uc) * 0.5f;
      Vec2F uca = (uc + ua) * 0.5f;

      int base = (int)new_uvs.size(); // 6
      new_uvs.push_back(ua);
      new_uvs.push_back(ub);
      new_uvs.push_back(uc);
      new_uvs.push_back(uab);
      new_uvs.push_back(ubc);
      new_uvs.push_back(uca);

      new_faces_uvs.push_back({base + 0, base + 3, base + 5}); // a,  ab, ca
      new_faces_uvs.push_back({base + 3, base + 1, base + 4}); // ab, b,  bc
      new_faces_uvs.push_back({base + 5, base + 4, base + 2}); // ca, bc, c
      new_faces_uvs.push_back({base + 3, base + 4, base + 5}); // ab, bc, ca
    }
  }

  m_faces.swap(new_faces);

  if (has_uv) {
    m_uvs.swap(new_uvs);
    m_faces_uvs.swap(new_faces_uvs);
  }
}