#pragma once
typedef struct {
  float m[9];
  float &at(int i, int j) { return m[i * 3 + j]; }
} Mat3;

typedef struct {
  float m[16];
  float &at(int i, int j) { return m[i * 4 + j]; }
} Mat4;

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

typedef struct {
  int m[3];
  int &at(int i) { return m[i]; }
} Vec3I;