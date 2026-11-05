typedef struct {
  float m[9];
  float &at(int i, int j) { return m[i * 3 + j]; }
} Mat3;

typedef struct {
  float m[16];
  float &at(int i, int j) { return m[i * 4 + j]; }
} Mat4;

typedef struct {
  float m[3];
  float &at(int i) { return m[i]; }
} Vec3F;

typedef struct {
  int m[3];
  int &at(int i) { return m[i]; }
} Vec3I;