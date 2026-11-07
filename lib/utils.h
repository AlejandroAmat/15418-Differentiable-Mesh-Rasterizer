#pragma once
#include <cmath>
#include <iostream>

#define WIDTH 400
#define HEIGHT 400
#define NUM_FACES_START 3000
#define BATCH_SIZE 21

struct Mat3 {
  float m[9];
  float &at(int i, int j) { return m[i * 3 + j]; }
  float at(int i, int j) const { return m[i * 3 + j]; }
  static Mat3 identity() { return {1, 0, 0, 0, 1, 0, 0, 0, 1}; }
};

struct Vec2F {
  float m[2];
  float &at(int i) { return m[i]; }
  float at(int i) const { return m[i]; }
  Vec2F operator*(float s) const { return {m[0] * s, m[1] * s}; }
  Vec2F operator+(const Vec2F &o) const {
    return {m[0] + o.m[0], m[1] + o.m[1]};
  }
  Vec2F operator-(const Vec2F &o) const {
    return {m[0] - o.m[0], m[1] - o.m[1]};
  }
  Vec2F &operator+=(const Vec2F &o) {
    m[0] += o.m[0];
    m[1] += o.m[1];
    return *this;
  }
  float x() const { return m[0]; }
  float y() const { return m[1]; }
  float dot(const Vec2F &o) const { return m[0] * o.m[0] + m[1] * o.m[1]; }
};

struct Vec3F {
  float m[3];
  float &at(int i) { return m[i]; }
  float at(int i) const { return m[i]; }
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
  float x() const { return m[0]; }
  float y() const { return m[1]; }
  float z() const { return m[2]; }
  Vec2F xy() const { return {m[0], m[1]}; }
  Vec3F operator-(const Vec3F &o) const {
    return {m[0] - o.m[0], m[1] - o.m[1], m[2] - o.m[2]};
  }
  float dot(const Vec3F &o) const {
    return m[0] * o.m[0] + m[1] * o.m[1] + m[2] * o.m[2];
  }
  float norm() const { return std::sqrt(dot(*this)); }
  Vec3F normalized() const { return *this * (1.0f / norm()); }
  Vec3F cross(const Vec3F &o) const {
    return {m[1] * o.m[2] - m[2] * o.m[1], m[2] * o.m[0] - m[0] * o.m[2],
            m[0] * o.m[1] - m[1] * o.m[0]};
  }
  void print() { printf("x: %f, y:%f, z:%f", m[0], m[1], m[2]); }
};

inline void toPixelSpace(const Mat3 &K, Vec3F &p) {
  p.at(0) = (K.at(0, 0) * p.at(0)) / p.at(2) + K.at(0, 2);
  p.at(1) = (K.at(1, 1) * p.at(1)) / p.at(2) + K.at(1, 2);
}

struct Mat4 {
  float m[16];
  float &at(int i, int j) { return m[i * 4 + j]; }
  float at(int i, int j) const { return m[i * 4 + j]; }
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
  int at(int i) const { return m[i]; }
} Vec3I;