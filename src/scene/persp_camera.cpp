#include "persp_camera.h"

void PerspectiveCamera::look_at(Vec3F center) {
  Vec3F forward = (center - pos).normalized();
  Vec3F right = forward.cross(up_world).normalized();
  Vec3F down = forward.cross(right);

  m_T = Mat4{right.x(),   right.y(),   right.z(),   -right.dot(pos),
             down.x(),    down.y(),    down.z(),    -down.dot(pos),
             forward.x(), forward.y(), forward.z(), -forward.dot(pos),
             0.0f,        0.0f,        0.0f,        1.0f};
}

void PerspectiveCamera::update_pos(int i, int n_views, float radius) {
  const int n_h = n_views / 2;
  const int n_v = n_views - n_h;

  Vec3F offset;
  if (i < n_h) {
    float az = 2.0f * (float)M_PI * i / n_h;
    offset = Vec3F{sinf(az), 0.0f, cosf(az)};
  } else {
    float t = 2.0f * (float)M_PI * (i - n_h) / n_v;
    offset = Vec3F{0.0f, sinf(t), cosf(t)};
  }
  pos = pos + offset * radius;
}