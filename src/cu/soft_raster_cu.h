#pragma once
#include "image.h"
#include "mesh.h"
#include "persp_camera.h"
#include "structs_cu.h"
#include "texture.h"
#include "utils.h"
#include <vector>

class SoftRasterizerCUDA {

public:
  SoftRasterizerCUDA(int batch = 1){};

  void add_cameras(const std::vector<CameraCUDA> &cams);
  void cuda_free();
};