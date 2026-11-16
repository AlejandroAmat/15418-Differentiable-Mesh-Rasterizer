#pragma once
#include "image.h"
#include "mesh.h"
#include "persp_camera.h"
#include "texture.h"
#include "utils.h"
#include <vector>

class SoftRasterizerCUDA {

public:
  SoftRasterizerCUDA(int w, int h, int batch = 1);
  ~SoftRasterizerCUDA();

  void cuda_init(std::vector<Mat3> &K, std::vector<Mat4> &Ts,
                 std::vector<Vec3I> &faces, std::vector<Vec2F> &uv,
                 std::vector<Vec3I> &faces_uvs, int verts, int tex_w, int tex_h,
                 int batch_size);

  void cuda_free();
};