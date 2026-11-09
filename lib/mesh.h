#pragma once
#include "utils.h"
#include <algorithm>
#include <map>
#include <utility>
#include <vector>

class Mesh {
public:
  Mesh() {}
  Mesh(const std::vector<Vec3I> &faces, const std::vector<Vec3F> &vertices)
      : m_faces(faces), m_vertices(vertices), m_num_verts(vertices.size()) {}

  void subdivide_midpoint();
  void update_mesh(const std::vector<Vec3I> &faces,
                   const std::vector<Vec3F> &vertices) {
    m_vertices = vertices;
    m_faces = faces;
    m_num_verts = vertices.size();
  }
  const std::vector<Vec3F> &vertices() const { return m_vertices; }
  std::vector<Vec3F> &vertices() { return m_vertices; }

  const std::vector<Vec3I> &faces() const { return m_faces; }
  std::vector<Vec3I> &faces() { return m_faces; }

  const int num_verts() const { return m_num_verts; };

  Vec3F get_center();
  Vec3F get_center_bbox(float &scale);

private:
  std::vector<Vec3I> m_faces;
  std::vector<Vec3F> m_vertices;
  int m_num_verts;
};