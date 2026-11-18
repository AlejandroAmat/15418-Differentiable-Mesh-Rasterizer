#include "soft_raster_cu.h"
#include <cub/cub.cuh>

#define LOAD_PHASE (BLOCK_SIZE / 2)

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

__device__ __forceinline__ float dot(float2 a, float2 b) {
  return a.x * b.x + a.y * b.y;
}
__device__ __forceinline__ float2 sub(float2 a, float2 b) {
  return {a.x - b.x, a.y - b.y};
}
__device__ __forceinline__ float2 mul(float2 a, float s) {
  return {a.x * s, a.y * s};
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

  float3 _p0 = to_pixel(view, v0);
  float3 _p1 = to_pixel(view, v1);
  float3 p2 = to_pixel(view, v2);

  // will not happen in our setup
  if (_p0.z <= 0 || _p1.z <= 0 || p2.z <= 0) {
    tiles_touched[global_id] = 0;
    return;
  }

  float2 p0 = {_p0.x, _p0.y};
  float2 p1 = {_p1.x, _p1.y};
  _face_data.p2 = {p2.x, p2.y};
  _face_data.iz0 = _p0.z;
  _face_data.iz1 = _p1.z;
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

  _face_data.e0 = sub(p0, _face_data.p2);
  _face_data.e1 = sub(p1, _face_data.p2);

  _face_data.e0_inv = 1.0f / dot(_face_data.e0, _face_data.e0);
  _face_data.e1_inv = 1.0f / dot(_face_data.e1, _face_data.e1);

  float2 e0e1 = sub(_face_data.e1, _face_data.e0);

  _face_data.e0e1_inv = 1.0f / dot(e0e1, e0e1);

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

__global__ void identify_ranges(int *keys, int *ranges, int total_faces_pairs) {
  int id_pair = blockIdx.x * blockDim.x + threadIdx.x;

  if (id_pair >= total_faces_pairs)
    return;
  int key =
      keys[id_pair]; // this is just the tile_view_id. we need different as same
                     // tile form diff views processes diff triangles

  if (id_pair == 0)
    ranges[key * 2] = 0;

  else {
    int prev = keys[id_pair - 1];
    if (prev != key) {
      ranges[prev * 2 + 1] = id_pair;
      ranges[key * 2] = id_pair;
    }
  }
  if (id_pair == total_faces_pairs - 1) {
    ranges[key * 2 + 1] = total_faces_pairs;
  }
}

__global__ void render(int *ranges, int *faces_sorted, FaceData *face_data,
                       RasterParamsDevice params, float *render_targets,
                       float *accums, float *lmax, float *wsum, int num_faces,
                       float *texture, int *faces_uvs, float *uvs, int text_w,
                       int text_h, short *bbox) {

  int view_id = blockIdx.z;
  int thread_id = threadIdx.y * BLOCK_X + threadIdx.x;
  int px = threadIdx.x + blockIdx.x * BLOCK_X;
  int py = threadIdx.y + blockIdx.y * BLOCK_Y;
  int global_pix = params.w * params.h * view_id + py * params.w + px;
  int key_id =
      view_id * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

  float2 pix_center = make_float2(px + 0.5, py + 0.5);

  int start = ranges[key_id * 2];
  int end = ranges[key_id * 2 + 1];

  bool only_loads_shared = (py >= params.h || px >= params.w);

  __shared__ FaceData s_face[LOAD_PHASE];
  __shared__ int s_face_id[LOAD_PHASE];
  __shared__ short4 s_bbox[LOAD_PHASE];
  // __shared__ float2 s_uv[BLOCK_X * BLOCK_Y][3];

  float4 *render_ptr = (float4 *)(&render_targets[global_pix * 4]);

  float margin_sq = params.sigma * logf((1.0f - EPSILON) / EPSILON);
  float accum = 1.0f;

  while (start < end) {

    int cur_end = min(start + LOAD_PHASE, end);

    __syncthreads();

    if (thread_id < cur_end - start) {
      s_face_id[thread_id] = faces_sorted[start + thread_id];
      s_face[thread_id] = face_data[view_id * num_faces + s_face_id[thread_id]];
      s_bbox[thread_id] =
          *(const short4
                *)(&bbox[(view_id * num_faces + s_face_id[thread_id]) * 4]);
    }

    __syncthreads();

    if (!only_loads_shared) {
      for (int i = 0; i < cur_end - start; i++) {

        short4 b = s_bbox[i];
        if (px < b.x || px >= b.y || py < b.z || py >= b.w)
          continue;

        const float2 &p2 = s_face[i].p2;
        float2 q = make_float2(pix_center.x - p2.x, pix_center.y - p2.y);

        const FaceData &f = s_face[i];

        float b0 = (q.x * f.e1.y - f.e1.x * q.y) * f.inv_d;
        float b1 = (f.e0.x * q.y - q.x * f.e0.y) * f.inv_d;
        float b2 = 1.0f - b0 - b1;
        int delta = (b0 > 0 && b1 > 0 && b2 > 0) ? 1 : -1;

        float2 e1e0 = sub(f.e1, f.e0);

        float t1 = fminf(fmaxf(dot(q, f.e0) * f.e0_inv, 0.0f), 1.0f);
        float t2 = fminf(fmaxf(dot(q, f.e1) * f.e1_inv, 0.0f), 1.0f);
        float t3 =
            fminf(fmaxf(dot(sub(q, f.e0), e1e0) * f.e0e1_inv, 0.0f), 1.0f);

        float2 dir1 = sub(q, mul(f.e0, t1));
        float2 dir2 = sub(q, mul(f.e1, t2));
        float2 dir3 = sub(sub(q, f.e0), mul(e1e0, t3));

        float min_dist =
            fminf(dot(dir1, dir1), fminf(dot(dir2, dir2), dot(dir3, dir3)));

        if (delta < 0 && min_dist > margin_sq)
          continue;

        float D = 1.0f / (1.0f + expf(-delta * min_dist * params.inv_sigma));
        accum *= (1.0f - D);
      }
    }
    start += LOAD_PHASE;
  }

  if (!only_loads_shared) {
    *render_ptr = make_float4(1.0f, 1.0f, 1.0f, 1.0f - accum);
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

  int blocks_pairs = (total_faces_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
  identify_ranges<<<blocks_pairs, BLOCK_SIZE>>>(
      m_pair_keys_sorted.ptr, m_ranges.ptr, total_faces_pairs);

  ////------------------------------///
  render<<<grid_tiled, block_tiled>>>(
      m_ranges.ptr, m_pair_faces_sorted.ptr, m_faces_data.ptr, m_params,
      m_render_targets.ptr, m_accums.ptr, m_lmax.ptr, m_wsum.ptr, faces,
      tex.data.ptr, mesh.faces_uvs.ptr, mesh.uvs.ptr, tex.w, tex.h, m_bbox.ptr);
  ////------------------------------------////
  // views are rendered
}