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

// LLM-generated
void load_obj(const std::string &file_in, Mesh &m) {
  std::ifstream f(file_in);
  if (!f)
    throw std::runtime_error("cannot open " + file_in);

  auto &verts = m.vertices();
  auto &faces = m.faces();
  auto &uvs = m.uvs();
  auto &faces_uvs = m.faces_uvs();
  verts.clear();
  faces.clear();
  uvs.clear();
  faces_uvs.clear();

  // OBJ indices are 1-based, negative = relative to end; -1 = missing
  auto fix = [](int idx, size_t n) {
    return (idx < 0) ? (int)n + idx : idx - 1;
  };

  std::string line;
  std::vector<int> poly, poly_uv;
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    std::string tag;
    ss >> tag;

    if (tag == "v") {
      float x, y, z;
      ss >> x >> y >> z;
      verts.push_back({x, y, z});
    } else if (tag == "vt") {
      float u, v;
      ss >> u >> v;
      uvs.push_back({u, 1.0f - v}); // flip so v=0 is the top image row
    } else if (tag == "f") {
      poly.clear();
      poly_uv.clear();
      std::string tok;
      while (ss >> tok) {
        // tok: "v", "v/vt", "v//vn" or "v/vt/vn"
        size_t s1 = tok.find('/');
        poly.push_back(fix(std::stoi(tok.substr(0, s1)), verts.size()));

        int ti = -1;
        if (s1 != std::string::npos && s1 + 1 < tok.size() &&
            tok[s1 + 1] != '/') {
          size_t s2 = tok.find('/', s1 + 1);
          ti = fix(std::stoi(tok.substr(s1 + 1, s2 - s1 - 1)), uvs.size());
        }
        poly_uv.push_back(ti);
      }
      for (size_t k = 1; k + 1 < poly.size(); ++k) {
        faces.push_back({poly[0], poly[k], poly[k + 1]});
        faces_uvs.push_back({poly_uv[0], poly_uv[k], poly_uv[k + 1]});
      }
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

// LLM generated
void compute_sphere_uvs(Mesh &m, const Vec3F &center) {
  const auto &V = m.vertices();
  const auto &F = m.faces();
  auto &uvs = m.uvs();
  auto &fuv = m.faces_uvs();
  uvs.clear();
  fuv.clear();
  uvs.reserve(3 * F.size());
  fuv.reserve(F.size());

  for (size_t t = 0; t < F.size(); ++t) {
    float u[3], v[3];
    bool pole[3];

    // (1) equirectangular: direction -> (longitude, latitude)
    for (int k = 0; k < 3; ++k) {
      Vec3F d = (V[F[t].at(k)] - center).normalized();
      u[k] = 0.5f + std::atan2(d.z(), d.x()) / (2.0f * (float)M_PI);
      v[k] = 0.5f - std::asin(std::clamp(d.y(), -1.0f, 1.0f)) / (float)M_PI;
      pole[k] = std::abs(d.y()) > 0.9999f;
    }

    // (2) seam: if corners straddle u=1 -> u=0, shift the small ones by +1
    float umax = -1e9f;
    for (int k = 0; k < 3; ++k)
      if (!pole[k])
        umax = std::max(umax, u[k]);
    for (int k = 0; k < 3; ++k)
      if (!pole[k] && umax - u[k] > 0.5f)
        u[k] += 1.0f;

    // (3) poles: longitude is undefined there, use the other corners' average
    for (int k = 0; k < 3; ++k) {
      if (!pole[k])
        continue;
      float s = 0;
      int n = 0;
      for (int j = 0; j < 3; ++j)
        if (!pole[j]) {
          s += u[j];
          n++;
        }
      if (n > 0)
        u[k] = s / n;
    }

    // (4) every corner gets its own UV entry
    int base = (int)uvs.size();
    for (int k = 0; k < 3; ++k)
      uvs.push_back({u[k], v[k]});
    fuv.push_back({base, base + 1, base + 2});
  }
}

std::vector<std::vector<int>> build_adjacency(const Mesh &m) {
  std::vector<std::vector<int>> nbrs(m.vertices().size());
  auto add = [&](int a, int b) {
    if (std::find(nbrs[a].begin(), nbrs[a].end(), b) == nbrs[a].end())
      nbrs[a].push_back(b);
  };
  for (const auto &f : m.faces()) {
    for (int k = 0; k < 3; k++) {
      int a = f.at(k), b = f.at((k + 1) % 3);
      add(a, b);
      add(b, a);
    }
  }
  return nbrs;
}