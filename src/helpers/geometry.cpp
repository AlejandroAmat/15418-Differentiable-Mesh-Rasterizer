#include "geometry.h"

void create_cube(Mesh &m, Vec3F center, float scale) {
  std::vector<Vec3F> cube_vertices = {
      {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
      {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1},
  };
  for (const Vec3F &u : cube_vertices)
    cube_vertices.push_back(center + u * (scale / 2));

  std::vector<Vec3I> cube_faces = {
      {4, 5, 6}, {4, 6, 7}, // front
      {0, 3, 2}, {0, 2, 1}, // back
      {1, 2, 6}, {1, 6, 5}, // right
      {0, 4, 7}, {0, 7, 3}, // left
      {3, 7, 6}, {3, 6, 2}, // top
      {0, 1, 5}, {0, 5, 4}, // bottom
  };

  m.update_mesh(cube_faces, cube_vertices);
}