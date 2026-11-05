#pragma once

struct Mat3 {
  float m[9];
  float &at(int i, int j) { return m[i * 3 + j]; }
  static Mat3 identity() { return {1, 0, 0, 0, 1, 0, 0, 0, 1}; }
};

struct Vec3F {
  float m[3];
  float &at(int i) { return m[i]; }
  Vec3F operator*(float s) const { return {m[0] * s, m[1] * s, m[2] * s}; }
  Vec3F operator+(const Vec3F &o) const {
    return {m[0] + o.m[0], m[1] + o.m[1], m[2] + o.m[2]};
  }
  Vec3F &operator+=(const Vec3F &o) {
    m[0] += o.m[0];
    m[1] += o.m[1];
    m[2] += o.m[2];
    return *this;
  }
};

struct Mat4 {
  float m[16];
  float &at(int i, int j) { return m[i * 4 + j]; }
  static Mat4 identity() {
    return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  }
  Vec3F operator*(const Vec3F &v) const {
    return {m[0] * v.m[0] + m[1] * v.m[1] + m[2] * v.m[2] + m[3],
            m[4] * v.m[0] + m[5] * v.m[1] + m[6] * v.m[2] + m[7],
            m[8] * v.m[0] + m[9] * v.m[1] + m[10] * v.m[2] + m[11]};
  }
};

typedef struct {
  int m[3];
  int &at(int i) { return m[i]; }
} Vec3I;