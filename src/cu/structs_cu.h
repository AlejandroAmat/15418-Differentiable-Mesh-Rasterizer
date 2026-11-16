#pragma once
#include "device_buffer.h"
#include "mesh.h"
#include "texture.h"
#include "utils.h"

struct MeshCUDA {

  DeviceBuffer<float> verts;
  DeviceBuffer<float> grads;
  DeviceBuffer<int> faces;
  DeviceBuffer<int> faces_uvs;
  DeviceBuffer<float> uvs;
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
  }
};

struct TextureGPU {

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