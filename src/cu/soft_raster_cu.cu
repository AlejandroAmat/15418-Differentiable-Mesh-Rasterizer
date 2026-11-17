#include "soft_raster_cu.h"
#include <cub/cub.cuh>

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
  m_tiles_touched.resize(m_batch * num_faces);

  m_bbox.resize(num_faces * m_batch * 4);

  // for cub:
  cub::DeviceScan::InclusiveSum(nullptr, m_scan_bytes, m_tiles_touched.ptr,
                                m_tiles_touched.ptr, m_batch * num_faces);
  m_temp_cub.resize(m_scan_bytes);

  int tiles_x = (m_w + BLOCK_X - 1) / BLOCK_X;
  int tiles_y = (m_h + BLOCK_Y - 1) / BLOCK_Y;
  m_ranges.resize(tiles_x * tiles_y * 2 * m_batch);
}

__device__ __forceinline__ float3 to_pixel(const CameraCUDA &c, float3 v) {
  float3 v_cam;
  float3 ret;
  v_cam.x = v.x * c.R[0] + v.y * c.R[1] + v.z * c.R[2] + c.t[0];
  v_cam.y = v.x * c.R[3] + v.y * c.R[4] + v.z * c.R[5] + c.t[1];
  v_cam.z = v.x * c.R[6] + v.y * c.R[7] + v.z * c.R[8] + c.t[2];

  float iz = 1.0f / v_cam.z;

  ret.x = c.fx * v_cam.x * iz + c.cx;
  ret.y = c.fy * v_cam.y * iz + c.cy;
  ret.z = iz;
  return ret;
}

__global__ void preprocess_faces(const float *verts, const int *faces,
                                 const int num_faces,
                                 RasterParamsDevice rast_params,
                                 FaceData *face_data, int *tiles_touched,
                                 short *bbox) {

  int face_id = blockIdx.x * blockDim.x + threadIdx.x;
  int view_id = blockIdx.y;
  int global_id = face_id + view_id * num_faces;
  FaceData _face_data;

  if (face_id >= num_faces)
    return;

  const CameraCUDA &view = device_cams[rast_params.views[view_id]];

  int3 F = *(int3 *)(&faces[face_id * 3]);

  float3 v0 = *(float3 *)(&verts[F.x * 3]);
  float3 v1 = *(float3 *)(&verts[F.y * 3]);
  float3 v2 = *(float3 *)(&verts[F.z * 3]);

  float3 p0 = to_pixel(view, v0);
  float3 p1 = to_pixel(view, v1);
  float3 p2 = to_pixel(view, v2);

  // will not happen in our setup
  if (p0.z <= 0 || p1.z <= 0 || p2.z <= 0) {
    tiles_touched[global_id] = 0;
    return;
  }

  _face_data.p0 = {p0.x, p0.y};
  _face_data.p1 = {p1.x, p1.y};
  _face_data.p2 = {p2.x, p2.y};
  _face_data.iz0 = p0.z;
  _face_data.iz1 = p1.z;
  _face_data.iz2 = p2.z;

  float bbox_margin =
      sqrtf(rast_params.sigma * logf((1.0f - EPSILON) / EPSILON));

  float min_x = fminf(p0.x, fminf(p1.x, p2.x));
  float max_x = fmaxf(p0.x, fmaxf(p1.x, p2.x));
  float min_y = fminf(p0.y, fminf(p1.y, p2.y));
  float max_y = fmaxf(p0.y, fmaxf(p1.y, p2.y));

  short x0 = (short)max(0, (int)floorf(min_x - bbox_margin));
  short x1 = (short)min(rast_params.w, (int)ceilf(max_x + bbox_margin) + 1);
  short y0 = (short)max(0, (int)floorf(min_y - bbox_margin));
  short y1 = (short)min(rast_params.h, (int)ceilf(max_y + bbox_margin) + 1);

  // will not happen in our setup
  if (max_x < 0 || max_y < 0 || min_x >= rast_params.w ||
      min_y >= rast_params.h) {
    tiles_touched[global_id] = 0;
    return;
  }

  _face_data.e0 = {_face_data.p0.x - _face_data.p2.x,
                   _face_data.p0.y - _face_data.p2.y};
  _face_data.e1 = {_face_data.p1.x - _face_data.p2.x,
                   _face_data.p1.y - _face_data.p2.y};
  float d =
      _face_data.e0.x * _face_data.e1.y - _face_data.e1.x * _face_data.e0.y;

  if (fabsf(d) < 1e-8) {
    tiles_touched[global_id] = 0;
    return;
  }

  _face_data.inv_d = 1.0f / d;

  int t_min_x = x0 / BLOCK_X;
  int t_max_x = (x1 + BLOCK_X - 1) / BLOCK_X;
  int t_min_y = y0 / BLOCK_Y;
  int t_max_y = (y1 + BLOCK_Y - 1) / BLOCK_Y;

  tiles_touched[face_id + view_id * num_faces] =
      (t_max_x - t_min_x) * (t_max_y - t_min_y);

  bbox[global_id * 4 + 0] = x0;
  bbox[global_id * 4 + 1] = x1;
  bbox[global_id * 4 + 2] = y0;
  bbox[global_id * 4 + 3] = y1;

  face_data[global_id] = _face_data;
}

__global__ void assign_pairs(int num_faces, int *tiles_touched, short *bbox,
                             dim3 grid, int *pair_keys, int *pair_faces) {

  int face_id = blockIdx.x * blockDim.x + threadIdx.x;
  int view_id = blockIdx.y;
  int global_id = face_id + view_id * num_faces;

  if (face_id >= num_faces)
    return;

  int start = global_id ? tiles_touched[global_id - 1] : 0;
  int end = tiles_touched[global_id];

  if (end == start)
    return;

  int x0 = bbox[global_id * 4 + 0];
  int x1 = bbox[global_id * 4 + 1];
  int y0 = bbox[global_id * 4 + 2];
  int y1 = bbox[global_id * 4 + 3];

  int t_min_x = x0 / BLOCK_X;
  int t_max_x = (x1 + BLOCK_X - 1) / BLOCK_X;
  int t_min_y = y0 / BLOCK_Y;
  int t_max_y = (y1 + BLOCK_Y - 1) / BLOCK_Y;

  for (int i = t_min_x; i < t_max_x; i++) {
    for (int j = t_min_y; j < t_max_y; j++) {

      int tile_id = j * grid.x + i;
      int tile_view_id = tile_id + grid.x * grid.y * view_id;

      pair_faces[start] = face_id;
      pair_keys[start] = tile_view_id;

      start++;
    }
  }
}

void SoftRasterizerCUDA::forward(MeshCUDA &mesh, TextureCUDA &tex,
                                 std::vector<int> &index, int start) {

  int faces = mesh.n_faces;
  // mesh.faces.resize() & restart accum
  for (int i = 0; i < m_batch; i++) {
    m_params.views[i] = index[start + i];
  }

  dim3 block_preprocess(BLOCK_SIZE);
  dim3 grid_preprocess((faces + BLOCK_SIZE - 1) / BLOCK_SIZE, m_batch);
  preprocess_faces<<<grid_preprocess, block_preprocess>>>(
      mesh.verts.ptr, mesh.faces.ptr, faces, m_params, m_faces_data.ptr,
      m_tiles_touched.ptr, m_bbox.ptr);

  //--------------------------------------//
  int N = m_batch * faces;
  cub::DeviceScan::InclusiveSum(m_temp_cub.ptr, m_scan_bytes,
                                m_tiles_touched.ptr, m_tiles_touched.ptr, N);

  int total_faces_pairs;
  m_tiles_touched.copy_from_device(&total_faces_pairs, N - 1, 1);
  //-------------------------------------//

  m_ranges.set_zero();

  m_pair_faces.reserve(total_faces_pairs);
  m_pair_faces_sorted.reserve(total_faces_pairs);
  m_pair_keys.reserve(total_faces_pairs);
  m_pair_keys_sorted.reserve(total_faces_pairs);

  int tiles_x = (m_w + BLOCK_X - 1) / BLOCK_X;
  int tiles_y = (m_h + BLOCK_Y - 1) / BLOCK_Y;
  dim3 block_tiled(BLOCK_X, BLOCK_Y);
  dim3 grid_tiled(tiles_x, tiles_y, m_batch);

  assign_pairs<<<grid_preprocess, block_preprocess>>>(
      faces, m_tiles_touched.ptr, m_bbox.ptr, grid_tiled, m_pair_keys.ptr,
      m_pair_faces.ptr);

  int key_bit =
      32 -
      __builtin_clz(m_batch * tiles_x *
                    tiles_y); // radix does passses in 8 bits. Having 400x400 at
                              // batch size 20 the higher bit is at 14. could
                              // hardcode but we might change resolution.

  cub::DeviceRadixSort::SortPairs(
      nullptr, m_sort_bytes, m_pair_keys.ptr, m_pair_keys_sorted.ptr,
      m_pair_faces.ptr, m_pair_faces_sorted.ptr, total_faces_pairs, 0, key_bit);

  m_temp_cub.reserve(std::max(m_sort_bytes, m_scan_bytes));

  cub::DeviceRadixSort::SortPairs(
      m_temp_cub.ptr, m_sort_bytes, m_pair_keys.ptr, m_pair_keys_sorted.ptr,
      m_pair_faces.ptr, m_pair_faces_sorted.ptr, total_faces_pairs, 0, key_bit);
}