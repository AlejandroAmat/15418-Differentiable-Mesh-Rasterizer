#include "soft_raster.h"
#include <iostream>
#include <vector>

void SoftRasterizerCPU::forward(Mesh &m, const Texture &tex, int view_idx) {

  // Project

  size_t num_verts = m.vertices().size();
  size_t num_faces = m.faces().size();

  projected_verts.resize(num_verts);
  faces.resize(num_faces);
  accum.reset(m_w, m_h);

  bool has_uv = !m.uvs().empty();

  PerspectiveCamera &cam = m_cameras[view_idx];

  size_t np = (size_t)m_w * m_h;
  float bg_logit = eps_bg / gamma;
  m_lmax.assign(np, bg_logit);
  m_wsum.assign(np, 0.0f);
  m_csum.assign(3 * np, 0.0f);

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

    if (max_x < 0 || max_y < 0 || min_x >= cam.get_w() ||
        min_y >= cam.get_h()) {
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
    Vec3F vx0 = projected_verts[cur_face.vert_idx.at(0)];
    Vec3F vx1 = projected_verts[cur_face.vert_idx.at(1)];

    float iz0 = 1.0f / vx0.z(), iz1 = 1.0f / vx1.z(), iz2 = 1.0f / vx2.z();

    Vec2F e0 = cur_face.e0;
    Vec2F e1 = cur_face.e1;
    Vec2F e1e0 = e1 - e0;
    Vec2F v2 = vx2.xy();

    bool face_uv =
        has_uv && i < m.faces_uvs().size() && m.faces_uvs()[i].at(0) >= 0;
    Vec2F uv0{0, 0}, uv1{0, 0}, uv2{0, 0};
    if (face_uv) {
      const Vec3I &fuv = m.faces_uvs()[i];
      uv0 = m.uvs()[fuv.at(0)];
      uv1 = m.uvs()[fuv.at(1)];
      uv2 = m.uvs()[fuv.at(2)];
    }

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

        // clamp barys
        float c0 = std::max(b0, 0.0f), c1 = std::max(b1, 0.0f),
              c2 = std::max(b2, 0.0f);
        float cs = c0 + c1 + c2;
        c0 /= cs;
        c1 /= cs;
        c2 /= cs;

        // depth score for this pixel
        float iz = c0 * iz0 + c1 * iz1 + c2 * iz2;
        float z = 1.0f / iz;
        float zn = std::clamp((z_far - z) / (z_far - z_near), 0.0f, 1.0f);

        // perspective correct barys
        float w0 = c0 * iz0 / iz, w1 = c1 * iz1 / iz, w2 = c2 * iz2 / iz;
        Vec3F col{1, 1, 1};
        if (face_uv) {
          Vec2F uv = uv0 * w0 + uv1 * w1 + uv2 * w2;
          col = tex.sample(uv.x(), uv.y());
        }

        size_t p = (size_t)y * m_w + x;
        float logit = zn / gamma;
        if (logit > m_lmax[p]) {
          float s = std::exp(m_lmax[p] - logit);
          m_wsum[p] *= s;
          for (int c = 0; c < 3; ++c)
            m_csum[3 * p + c] *= s;
          m_lmax[p] = logit;
        }

        float w = D * std::exp(logit - m_lmax[p]);
        m_wsum[p] += w;
        for (int c = 0; c < 3; ++c)
          m_csum[3 * p + c] += w * col.at(c);
      }
    }
  }

  float alpha;

  for (int y = 0; y < m_h; y++) {
    for (int x = 0; x < m_w; x++) {
      size_t p = (size_t)y * m_w + x;
      float wbg = std::exp(bg_logit - m_lmax[p]);
      float inv = 1.0f / (m_wsum[p] + wbg);
      for (int c = 0; c < 3; ++c)
        m_render_target.at(x, y, c) =
            (m_csum[3 * p + c] + wbg * bg_color.at(c)) * inv;
      m_render_target.at(x, y, 3) = 1.0f - accum.at(x, y);
    }
  }
}

void SoftRasterizerCPU::backward(LossData &loss, Mesh &mesh, const Texture &tex,
                                 int idx, std::vector<Vec3F> &grads,
                                 Image &tex_grad) {
  size_t num_verts = mesh.vertices().size();
  PerspectiveCamera &cam = m_cameras[idx];

  std::vector<float> gu(num_verts, 0.0f), gv(num_verts, 0.0f);

  bool has_uv = !mesh.uvs().empty();
  float bg_logit = eps_bg / gamma;

  float bbox_margin = std::sqrt(sigma * std::log((1.0f - epsilon) / epsilon));
  float margin_sq = bbox_margin * bbox_margin;

  for (size_t i = 0; i < faces.size(); ++i) {
    Face &f = faces[i];
    if (!f.shown)
      continue;

    Vec3F vx0 = projected_verts[f.vert_idx.at(0)];
    Vec3F vx1 = projected_verts[f.vert_idx.at(1)];
    Vec3F vx2 = projected_verts[f.vert_idx.at(2)];

    Vec2F v2 = vx2.xy();
    Vec2F e0 = f.e0, e1 = f.e1, e1e0 = e1 - e0;
    float inv_d = f.d;
    float e0_inv = 1.0f / e0.dot(e0);
    float e1_inv = 1.0f / e1.dot(e1);
    float e1e0_inv = 1.0f / e1e0.dot(e1e0);
    float iz0 = 1.0f / vx0.z(), iz1 = 1.0f / vx1.z(), iz2 = 1.0f / vx2.z();

    bool face_uv =
        has_uv && i < mesh.faces_uvs().size() && mesh.faces_uvs()[i].at(0) >= 0;
    Vec2F uv0{0, 0}, uv1{0, 0}, uv2{0, 0};
    if (face_uv) {
      const Vec3I &fuv = mesh.faces_uvs()[i];
      uv0 = mesh.uvs()[fuv.at(0)];
      uv1 = mesh.uvs()[fuv.at(1)];
      uv2 = mesh.uvs()[fuv.at(2)];
    }

    for (int y = f.bbox[2]; y <= f.bbox[3]; y++) {
      for (int x = f.bbox[0]; x <= f.bbox[1]; x++) {
        Vec2F q = Vec2F{x + 0.5f, y + 0.5f} - v2;

        float b0 = (q.x() * e1.y() - e1.x() * q.y()) * inv_d;
        float b1 = (e0.x() * q.y() - q.x() * e0.y()) * inv_d;
        float b2 = 1 - b0 - b1;
        float delta = (b0 > 0 && b1 > 0 && b2 > 0) ? 1.0f : -1.0f;

        float t1 = std::min(std::max(q.dot(e0) * e0_inv, 0.0f), 1.0f);
        float t2 = std::min(std::max(q.dot(e1) * e1_inv, 0.0f), 1.0f);
        float t3 =
            std::min(std::max((q - e0).dot(e1e0) * e1e0_inv, 0.0f), 1.0f);
        Vec2F dir1 = q - e0 * t1;
        Vec2F dir2 = q - e1 * t2;
        Vec2F dir3 = q - e0 - e1e0 * t3;
        float d1 = dir1.dot(dir1), d2 = dir2.dot(dir2), d3 = dir3.dot(dir3);
        float min_dist = std::min({d1, d2, d3});

        if (delta < 0 && min_dist > margin_sq)
          continue;

        float D = 1.0f / (1.0f + std::exp(-delta * min_dist * inv_sigma));

        // This is dL/dd^2
        float g =
            loss.loss_grad.at(x, y, 3) * accum.at(x, y) * D * delta * inv_sigma;

        int a, b;
        float t;
        Vec2F r;
        if (d1 <= d2 && d1 <= d3) {
          a = 2;
          b = 0;
          t = t1;
          r = dir1;
        } else if (d2 <= d3) {
          a = 2;
          b = 1;
          t = t2;
          r = dir2;
        } else {
          a = 0;
          b = 1;
          t = t3;
          r = dir3;
        }

        // dL/da.u , dL/da.v .. b
        int ia = f.vert_idx.at(a), ib = f.vert_idx.at(b);
        float s = -2.0f * g;
        gu[ia] += s * r.x() * (1 - t);
        gv[ia] += s * r.y() * (1 - t);
        gu[ib] += s * r.x() * t;
        gv[ib] += s * r.y() * t;

        // tex
        if (face_uv) {
          // recmoute uV
          float c0 = std::max(b0, 0.0f), c1 = std::max(b1, 0.0f),
                c2 = std::max(b2, 0.0f);
          float cs = c0 + c1 + c2;
          c0 /= cs;
          c1 /= cs;
          c2 /= cs;

          float iz = c0 * iz0 + c1 * iz1 + c2 * iz2;
          float z = 1.0f / iz;
          float zn = std::clamp((z_far - z) / (z_far - z_near), 0.0f, 1.0f);

          float w0 = c0 * iz0 / iz, w1 = c1 * iz1 / iz, w2 = c2 * iz2 / iz;
          Vec2F uv = uv0 * w0 + uv1 * w1 + uv2 * w2;

          size_t p = (size_t)y * m_w + x;
          float w = D * std::exp(zn / gamma - m_lmax[p]);

          // dpixel / dcol
          float W = m_wsum[p] + std::exp(bg_logit - m_lmax[p]);
          float share = w / W;

          // dL / dcol, then spread to the 4 texels
          BilinearSample bs = tex.bilinear(uv.x(), uv.y());
          for (int c = 0; c < 3; ++c) {
            float gc = loss.loss_grad.at(x, y, c) *
                       share; // dL/dcol_j,c (dL/Dpixel * dpixel/dcol)
            if (gc == 0.0f)
              continue;
            for (int k = 0; k < 4; ++k)
              tex_grad.at(bs.x[k], bs.y[k], c) += gc * bs.w[k];
          }
        }
      }
    }
  }

  float fx = cam.K().at(0, 0), cx = cam.K().at(0, 2);
  float fy = cam.K().at(1, 1), cy = cam.K().at(1, 2);
  const Mat4 &T = cam.T();

  for (size_t i = 0; i < num_verts; ++i) {
    if (gu[i] == 0.0f && gv[i] == 0.0f)
      continue;
    const Vec3F &p = projected_verts[i];
    float Z = p.z();

    float gX = gu[i] * fx / Z;
    float gY = gv[i] * fy / Z;
    float gZ = -(gu[i] * (p.x() - cx) + gv[i] * (p.y() - cy)) / Z;

    grads[i] += Vec3F{T.at(0, 0) * gX + T.at(1, 0) * gY + T.at(2, 0) * gZ,
                      T.at(0, 1) * gX + T.at(1, 1) * gY + T.at(2, 1) * gZ,
                      T.at(0, 2) * gX + T.at(1, 2) * gY + T.at(2, 2) * gZ};
  }
}

// LLM-generated to check correct behaviour of meshes
void SoftRasterizerCPU::render_shaded(Mesh &m, const Texture &tex, int view_idx,
                                      Image &out) {
  const auto &V = m.vertices();
  const auto &F = m.faces();
  const auto &UV = m.uvs();
  const auto &FUV = m.faces_uvs();
  bool has_uv = !UV.empty();
  PerspectiveCamera &cam = m_cameras[view_idx];

  // vertex normals: sum of unnormalized face normals (= area weighted)
  std::vector<Vec3F> N(V.size(), Vec3F{0, 0, 0});
  for (const auto &f : F) {
    const Vec3F &a = V[f.at(0)], &b = V[f.at(1)], &c = V[f.at(2)];
    Vec3F fn = (b - a).cross(c - a);
    N[f.at(0)] += fn;
    N[f.at(1)] += fn;
    N[f.at(2)] += fn;
  }
  for (auto &n : N) {
    float l = n.norm();
    if (l > 0)
      n = n * (1.0f / l);
  }

  const Mat4 &T = cam.T();
  Vec3F to_cam{-T.at(2, 0), -T.at(2, 1), -T.at(2, 2)};
  Vec3F light = (to_cam + Vec3F{0, 1, 0} * 0.7f).normalized();

  std::vector<Vec3F> P(V.size());
  for (size_t i = 0; i < V.size(); ++i) {
    P[i] = cam.T() * V[i];
    toPixelSpace(cam.K(), P[i]);
  }

  std::vector<float> zinv(m_w * m_h, 0.0f);
  for (int y = 0; y < m_h; y++)
    for (int x = 0; x < m_w; x++) {
      for (int c = 0; c < 3; c++)
        out.at(x, y, c) = 1.0f; // white
      out.at(x, y, 3) = 1.0f;   // opaque
    }

  for (size_t t = 0; t < F.size(); ++t) { // need t to index faces_uvs
    const Vec3I &f = F[t];
    const Vec3F &p0 = P[f.at(0)], &p1 = P[f.at(1)], &p2 = P[f.at(2)];
    if (p0.z() <= 0 || p1.z() <= 0 || p2.z() <= 0)
      continue;

    Vec2F e0 = p0.xy() - p2.xy(), e1 = p1.xy() - p2.xy();
    float d = e0.x() * e1.y() - e1.x() * e0.y();
    if (std::abs(d) < 1e-8f)
      continue;
    float inv_d = 1.0f / d;

    int x0 = std::max((int)std::floor(std::min({p0.x(), p1.x(), p2.x()})), 0);
    int x1 =
        std::min((int)std::ceil(std::max({p0.x(), p1.x(), p2.x()})), m_w - 1);
    int y0 = std::max((int)std::floor(std::min({p0.y(), p1.y(), p2.y()})), 0);
    int y1 =
        std::min((int)std::ceil(std::max({p0.y(), p1.y(), p2.y()})), m_h - 1);

    float iz0 = 1.0f / p0.z(), iz1 = 1.0f / p1.z(), iz2 = 1.0f / p2.z();

    // this face's 3 corner UVs (if it has them)
    bool face_uv =
        has_uv && t < m.faces_uvs().size() && m.faces_uvs()[t].at(0) >= 0;
    Vec2F uv0{0, 0}, uv1{0, 0}, uv2{0, 0};
    if (face_uv) {
      const Vec3I &fuv = m.faces_uvs()[t];
      uv0 = m.uvs()[fuv.at(0)];
      uv1 = m.uvs()[fuv.at(1)];
      uv2 = m.uvs()[fuv.at(2)];
    }
    for (int y = y0; y <= y1; y++) {
      for (int x = x0; x <= x1; x++) {
        Vec2F q = Vec2F{x + 0.5f, y + 0.5f} - p2.xy();
        float b0 = (q.x() * e1.y() - e1.x() * q.y()) * inv_d;
        float b1 = (e0.x() * q.y() - q.x() * e0.y()) * inv_d;
        float b2 = 1.0f - b0 - b1;
        if (b0 < 0 || b1 < 0 || b2 < 0)
          continue;

        float iz = b0 * iz0 + b1 * iz1 + b2 * iz2;
        float &zb = zinv[y * m_w + x];
        if (iz <= zb)
          continue;
        zb = iz;

        // perspective-correct weights (same ones for normals and UVs)
        float w0 = b0 * iz0 / iz, w1 = b1 * iz1 / iz, w2 = b2 * iz2 / iz;
        Vec3F n =
            (N[f.at(0)] * w0 + N[f.at(1)] * w1 + N[f.at(2)] * w2).normalized();
        float shade = 0.2f + 0.8f * std::abs(n.dot(light));

        Vec3F albedo{1, 1, 1};
        if (face_uv) {
          Vec2F uv = uv0 * w0 + uv1 * w1 + uv2 * w2;
          albedo = tex.sample(uv.x(), uv.y());
        }

        out.at(x, y, 0) = albedo.x() * shade;
        out.at(x, y, 1) = albedo.y() * shade;
        out.at(x, y, 2) = albedo.z() * shade;
        out.at(x, y, 3) = 1.0f;
      }
    }
  }
}