
constexpr float EPSILON = 1e-4f;
constexpr float GAMMA = 1e-4f;
constexpr float EPS_BG = 1e-3f;
constexpr float Z_NEAR = 0.1f;
constexpr float Z_FAR = 2.5f;
constexpr float BG_R = 1.0f; // always white so R is enough

struct __align__(16) FaceData {
  float2 p0, p1, p2;
  float iz0, iz1, iz2;
  float2 e0, e1;
  float inv_d;
};