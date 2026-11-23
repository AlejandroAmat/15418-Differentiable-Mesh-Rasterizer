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
  int num_views;
  int views[BATCH_SIZE];
  int w;
  int h;
  float bg_color = 1.0;
  float inv_n;
};

class SoftRasterizerCUDA {

public:
  SoftRasterizerCUDA(int w, int h, int batch = 1)
      : m_w(w), m_h(h), m_batch(batch) {
    m_params.w = w;
    m_params.h = h;
    m_params.num_views = batch;
    m_params.inv_n = 1.0f / (m_w * m_h * BATCH_SIZE);
  };

  void init_allocate(int total_views, int num_faces,
                     std::vector<Image> &gt_images, int num_verts);
  void add_cameras(const std::vector<CameraCUDA> &cams);
  void cuda_free(); // free all that data... wont be necessary with device
                    // bufefr implementation

  void forward(MeshCUDA &mesh, TextureCUDA &tex, std::vector<int> &index,
               int start);

  void backward(MeshCUDA &mesh, TextureCUDA &tex, std::vector<int> &index,
                int start, float w_rgb);

  void laplacian_reg(MeshCUDA &mesh, float lambda);

  void subdivide_midpoint(MeshCUDA &mesh);

  void set_sigma(float sigma) {
    m_params.sigma = sigma;
    m_params.inv_sigma = 1.0f / sigma;
  }

  void get_image(int num, Image &img) {
    size_t px = m_w * m_h * 4;
    m_render_targets.copy_from_device(&img.at(0, 0), num * px, px);
  }

private:
  int m_w, m_h, m_batch;
  DeviceBuffer<float> m_render_targets;
  DeviceBuffer<float> m_accums;
  DeviceBuffer<float> m_lmax;
  DeviceBuffer<float> m_wsum;
  DeviceBuffer<float> m_gt_images;
  DeviceBuffer<FaceData> m_faces_data;
  DeviceBuffer<int> m_tiles_touched;
  DeviceBuffer<short> m_bbox;

  DeviceBuffer<char> m_temp_cub;
  size_t m_scan_bytes = 0;
  size_t m_sort_bytes = 0;

  DeviceBuffer<int> m_pair_keys;
  DeviceBuffer<int> m_pair_faces;
  DeviceBuffer<int> m_pair_faces_sorted;
  DeviceBuffer<int> m_pair_keys_sorted;
  DeviceBuffer<int> m_ranges;

  DeviceBuffer<float> m_uv_grads;
  RasterParamsDevice m_params;
};