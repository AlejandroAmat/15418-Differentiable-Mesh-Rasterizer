#include "geometry.h"

void create_cube(Mesh &m, Vec3F center, float scale) {
  std::vector<Vec3F> cube_vertices = {
      {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
      {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1},
  };

  std::vector<Vec3F> verts;
  for (const Vec3F &u : cube_vertices)
    verts.push_back(center + u * (scale / 2));

  std::vector<Vec3I> cube_faces = {
      {4, 5, 6}, {4, 6, 7}, // front
      {0, 3, 2}, {0, 2, 1}, // back
      {1, 2, 6}, {1, 6, 5}, // right
      {0, 4, 7}, {0, 7, 3}, // left
      {3, 7, 6}, {3, 6, 2}, // top
      {0, 1, 5}, {0, 5, 4}, // bottom
  };

  m.update_mesh(cube_faces, verts);
}

void load_obj(const std::string &file_in, Mesh &m) {
  std::ifstream f(file_in);
  if (!f)
    throw std::runtime_error("cannot open " + file_in);

  auto &verts = m.vertices();
  auto &faces = m.faces();
  verts.clear();
  faces.clear();

  std::string line;
  std::vector<int> poly;
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    std::string tag;
    ss >> tag;

    if (tag == "v") {
      float x, y, z;
      ss >> x >> y >> z;
      verts.push_back({x, y, z});
    } else if (tag == "f") {
      poly.clear();
      std::string tok;
      while (ss >> tok) {
        int idx = std::stoi(tok.substr(0, tok.find('/')));
        idx = (idx < 0) ? (int)verts.size() + idx : idx - 1;
        poly.push_back(idx);
      }
      for (size_t k = 1; k + 1 < poly.size(); ++k)
        faces.push_back({poly[0], poly[k], poly[k + 1]});
    }
  }
}

void create_sphere(Mesh &m, Vec3F center, float scale, int num_faces) {
  int stacks =
      std::max(2, (int)std::round((1.0f + std::sqrt(1.0f + num_faces)) / 2.0f));
  int slices = 2 * stacks;
  float r = scale / 2;

  std::vector<Vec3F> verts;
  verts.push_back(center + Vec3F{0, r, 0});
  for (int i = 1; i < stacks; i++) {
    float th = (float)M_PI * i / stacks;
    for (int j = 0; j < slices; j++) {
      float ph = 2.0f * (float)M_PI * j / slices;
      verts.push_back(center + Vec3F{std::sin(th) * std::cos(ph), std::cos(th),
                                     std::sin(th) * std::sin(ph)} *
                                   r);
    }
  }
  verts.push_back(center + Vec3F{0, -r, 0});
  int bottom = (int)verts.size() - 1;

  auto ring = [&](int i, int j) { return 1 + (i - 1) * slices + (j % slices); };

  std::vector<Vec3I> faces;
  for (int j = 0; j < slices; j++)
    faces.push_back({0, ring(1, j + 1), ring(1, j)});
  for (int i = 1; i < stacks - 1; i++) {
    for (int j = 0; j < slices; j++) {
      int a = ring(i, j), b = ring(i, j + 1);
      int c = ring(i + 1, j), d = ring(i + 1, j + 1);
      faces.push_back({a, b, c});
      faces.push_back({b, d, c});
    }
  }
  for (int j = 0; j < slices; j++)
    faces.push_back({ring(stacks - 1, j), ring(stacks - 1, j + 1), bottom});

  m.update_mesh(faces, verts);
}