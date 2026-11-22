#pragma once
#include "device_buffer.h"
#include "geometry.h"
#include "mesh.h"
#include "persp_camera.h"
#include "texture.h"

#include "utils.h"

struct MeshCUDA {

  DeviceBuffer<float> verts;
  DeviceBuffer<float> grads;
  DeviceBuffer<int> faces;
  DeviceBuffer<int> faces_uvs;
  DeviceBuffer<float> uvs;
  DeviceBuffer<int> adj_offsets;
  DeviceBuffer<int> adj_idx;
  DeviceBuffer<float> lapl_delta;

  int n_verts = 0;
  int n_faces = 0;

  void upload_to_device(const Mesh &m) {
    n_verts = (int)m.vertices().size();
    n_faces = (int)m.faces().size();

    verts.upload_to_device(reinterpret_cast<const float *>(m.vertices().data()),
                           3 * n_verts);
    grads.resize(n_verts * 3);
    grads.set_zero();

    faces.upload_to_device(reinterpret_cast<const int *>(m.faces().data()),
                           3 * n_faces);

    uvs.upload_to_device(reinterpret_cast<const float *>(m.uvs().data()),
                         2 * m.uvs().size());

    faces_uvs.upload_to_device(
        reinterpret_cast<const int *>(m.faces_uvs().data()), 3 * n_faces);

    std::vector<std::vector<int>> adj = build_adjacency(m);
    std::vector<int> offsets(n_verts + 1);
    std::vector<int> idx;
    offsets[0] = 0;

    for (int v = 0; v < n_verts; v++) {
      idx.insert(idx.end(), adj[v].begin(), adj[v].end());
      offsets[v + 1] = offsets[v] + adj[v].size();
    }

    adj_offsets.upload_to_device(offsets.data(), n_verts + 1);
    adj_idx.upload_to_device(idx.data(), idx.size());
    lapl_delta.resize(n_verts * 3);
  }
};

struct TextureCUDA {

  DeviceBuffer<float> data;
  DeviceBuffer<float> grads;
  int w;
  int h;

  void upload_to_device(const Texture &tex) {
    w = tex.get_w();
    h = tex.get_h();

    grads.resize(w * h * 3);
    grads.set_zero();

    data.upload_to_device(tex.image().data(), w * h * 3);
  }

  void get_from_device(Texture &tex) {
    data.get_from_device(tex.image().data());
  }
};

struct CameraCUDA {
  float fx, fy, cx, cy;
  float R[9];
  float t[3];

  void from_cam(const PerspectiveCamera &cam) {
    fx = cam.K().at(0, 0);
    fy = cam.K().at(1, 1);
    cx = cam.K().at(0, 2);
    cy = cam.K().at(1, 2);

    const Mat4 &T = cam.T();
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        R[i * 3 + j] = T.at(i, j);
      }
      t[i] = T.at(i, 3); // last col
    }
  }
};