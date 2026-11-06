#include "soft_raster.h"
#include <iostream>
#include <vector>

void SoftRasterizerCPU::forward(PerspectiveCamera &cam, Mesh &m,
                                Image &render) {

  // Project

  size_t num_verts = m.vertices().size();
  size_t num_faces = m.faces().size();

  projected_verts.resize(num_verts);
  faces.resize(num_faces);
  accum.reset(m_w, m_h);

  for (size_t i = 0; i < num_verts; ++i) {
    Vec3F projected = cam.T() * m.vertices()[i];
    toPixelSpace(cam.K(), projected);
    projected_verts[i] = projected;
  }
  // projected now holds u,v and z;

  float bbox_margin = std::sqrt(sigma * std::log((1.0f - epsilon) / epsilon));

  for (size_t i = 0; i < num_faces; ++i) {
    Face &cur_face = faces[i];
    cur_face.shown = true;

    cur_face.vert_idx = m.faces()[i];

    Vec3F vx0 = projected_verts[cur_face.vert_idx.at(0)];
    Vec3F vx1 = projected_verts[cur_face.vert_idx.at(1)];
    Vec3F vx2 = projected_verts[cur_face.vert_idx.at(2)];

    if (vx0.z() <= 0 || vx1.z() <= 0 || vx2.z() <= 0) {
      cur_face.shown = false;
      continue;
    }
    float min_x = std::min({vx0.x(), vx1.x(), vx2.x()});
    float max_x = std::max({vx0.x(), vx1.x(), vx2.x()});
    float min_y = std::min({vx0.y(), vx1.y(), vx2.y()});
    float max_y = std::max({vx0.y(), vx1.y(), vx2.y()});

    if (min_x < 0 || min_y < 0 || max_x >= cam.get_w() ||
        max_y >= cam.get_h()) {
      cur_face.shown = false;
      continue;
    }

    cur_face.bbox[0] = std::max((int)std::floor(min_x - bbox_margin), 0);
    cur_face.bbox[1] =
        std::min((int)std::ceil(max_x + bbox_margin), cam.get_w() - 1);
    cur_face.bbox[2] = std::max((int)std::floor(min_y - bbox_margin), 0);
    cur_face.bbox[3] =
        std::min((int)std::ceil(max_y + bbox_margin), cam.get_h() - 1);

    // for barys
    cur_face.e0 = vx0.xy() - vx2.xy();
    cur_face.e1 = vx1.xy() - vx2.xy();
    cur_face.d =
        (cur_face.e0.x() * cur_face.e1.y() - cur_face.e1.x() * cur_face.e0.y());

    if (std::abs(cur_face.d) < 1e-8) {
      // degenerate
      cur_face.shown = false;
      continue;
    }
    cur_face.d = 1 / cur_face.d;
  }

  // render loop

  for (size_t i = 0; i < num_faces; ++i) {
    Face &cur_face = faces[i];
    if (!cur_face.shown) {
      continue;
    }

    Vec3F vx2 = projected_verts[cur_face.vert_idx.at(2)];

    Vec2F e0 = cur_face.e0;
    Vec2F e1 = cur_face.e1;
    Vec2F e1e0 = e1 - e0;
    Vec2F v2 = vx2.xy();

    float margin_sq = bbox_margin * bbox_margin;

    float inv_d = cur_face.d;

    float e0_inv = 1.0 / (e0.dot(e0));
    float e1_inv = 1.0 / (e1.dot(e1));
    float e1e0_inv = 1.0 / (e1e0).dot(e1e0);

    for (int y = cur_face.bbox[2]; y <= cur_face.bbox[3]; y++) {
      for (int x = cur_face.bbox[0]; x <= cur_face.bbox[1]; x++) {
        Vec2F pix_center = {x + 0.5f, y + 0.5f};

        Vec2F q = pix_center - v2;

        float b0 = (q.x() * e1.y() - e1.x() * q.y()) * inv_d;
        float b1 = (e0.x() * q.y() - q.x() * e0.y()) * inv_d;
        float b2 = 1 - b0 - b1;

        int delta = (b0 > 0 & b1 > 0 & b2 > 0) ? 1 : -1;

        float t1 = std::min(std::max(q.dot(e0) * e0_inv, 0.0f), 1.0f);
        float t2 = std::min(std::max(q.dot(e1) * e1_inv, 0.0f), 1.0f);
        float t3 =
            std::min(std::max((q - e0).dot(e1e0) * e1e0_inv, 0.0f), 1.0f);

        Vec2F dir1 = (q - e0 * t1);
        Vec2F dir2 = (q - e1 * t2);
        Vec2F dir3 = (q - e0 - e1e0 * t3);
        float d1 = dir1.dot(dir1);
        float d2 = dir2.dot(dir2);
        float d3 = dir3.dot(dir3);

        float min_dist = std::min({d1, d2, d3});

        if (delta < 0 && min_dist > margin_sq) {
          continue;
        }

        float D = 1.0f / (1.0f + std::exp(-delta * min_dist * inv_sigma));

        accum.at(x, y) *= (1 - D);
      }
    }
  }

  float alpha;
  for (int y = 0; y < m_h; y++) {
    for (int x = 0; x < m_w; x++) {
      alpha = 1 - accum.at(x, y);
      render.at(x, y) =
          std::round(std::min(std::max(255.0f * alpha, 0.0f), 255.0f));
    }
  }
}
