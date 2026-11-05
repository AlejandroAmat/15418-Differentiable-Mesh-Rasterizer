#pragma once
#include "utils.h"
#include <vector>

class Mesh {
public:
  Mesh() {}
  Mesh(const std::vector<Vec3I> &faces, const std::vector<Vec3F> &vertices)
      : m_faces(faces), m_vertices(vertices), m_num_verts(vertices.size()) {}

  void subdivide();
  void update_mesh(const std::vector<Vec3I> &faces,
                   const std::vector<Vec3F> &vertices) {
    m_vertices = vertices;
    m_faces = faces;
    m_num_verts = vertices.size();
  }
  const std::vector<Vec3F> &vertices() const { return m_vertices; }
  const std::vector<Vec3I> &faces() const { return m_faces; }
  const int num_verts() const { return m_num_verts; };

private:
  std::vector<Vec3I> m_faces;
  std::vector<Vec3F> m_vertices;
  int m_num_verts;
};