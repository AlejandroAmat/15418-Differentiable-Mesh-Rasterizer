#pragma once
#include "mesh.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

#include "utils.h"

void create_cube(Mesh &m, Vec3F center, float scale);

void load_obj(const std::string &file_in, Mesh &m);
void create_sphere(Mesh &m, Vec3F center, float scale, int num_faces);
void compute_sphere_uvs(Mesh &m, const Vec3F &center);
std::vector<std::vector<int>> build_adjacency(const Mesh &m);
