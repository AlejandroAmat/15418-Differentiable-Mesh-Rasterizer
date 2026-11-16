#include "soft_raster_cu.h"

__constant__ CameraCUDA device_cams[MAX_CAMS]; // we use 41 but just in case

void SoftRasterizerCUDA::add_cameras(const std::vector<CameraCUDA> &cams) {
  cudaMemcpyToSymbol(device_cams, cams.data(),
                     cams.size() * sizeof(CameraCUDA));
}

void SoftRasterizerCUDA::init_allocate(int total_views, int num_faces) {

  m_render_targets.resize(m_w * m_h * 4 * m_batch);
  m_gt_images.resize(m_w * m_h * 4 * total_views);

  m_accums.resize(m_w * m_h * m_batch);
  m_lmax.resize(m_w * m_h * m_batch);
  m_wsum.resize(m_w * m_h * m_batch);

  m_faces_data.resize(m_batch * num_faces);
}

void SoftRasterizerCUDA::forward(MeshCUDA &mesh, TextureCUDA &tex,
                                 std::vector<int> &index) {}