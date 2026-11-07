#pragma once
#include "mesh.h"
#include <fstream>
#include <sstream>
#include <string>

#include "utils.h"

void create_cube(Mesh &m, Vec3F center, float scale);

void load_obj(const std::string &file_in, Mesh &m);
void create_sphere(Mesh &m, Vec3F center, float scale, int num_faces);