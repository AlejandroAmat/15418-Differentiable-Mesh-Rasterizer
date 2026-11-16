#pragma once
#include "image.h"
#include "mesh.h"
#include "persp_camera.h"
#include "structs_cu.h"
#include "texture.h"
#include "utils.cuh"
#include "utils.h"
#include <vector>

struct RasterParamsDevice {
  float sigma;
  float inv_sigma;
};

class SoftRasterizerCUDA {

public:
  SoftRasterizerCUDA(int w, int h, int batch = 1)
      : m_w(w), m_h(h), m_batch(batch){};

  void init_allocate(int total_views, int num_faces);
  void add_cameras(const std::vector<CameraCUDA> &cams);
  void cuda_free(); // free all that data... wont be necessary with device
                    // bufefr implementation

  void set_sigma(float sigma) {
    m_params.sigma = sigma;
    m_params.inv_sigma = 1.0f / sigma;
  }

private:
  int m_w, m_h, m_batch;
  DeviceBuffer<float> m_render_targets;
  DeviceBuffer<float> m_accums;
  DeviceBuffer<float> m_lmax;
  DeviceBuffer<float> m_wsum;
  DeviceBuffer<float> m_gt_images;
  DeviceBuffer<FaceData> m_faces_data;

  RasterParamsDevice m_params;
};