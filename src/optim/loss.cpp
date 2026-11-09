#include "loss.h"

void L2(const Image &render, const Image &gt, LossData &loss) {
  double accum = 0;
  float pix_dif;
  float inv_N = 1.0 / (WIDTH * HEIGHT * BATCH_SIZE);
  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      pix_dif = render.at(x, y) - gt.at(x, y);
      accum += pix_dif * pix_dif;

      loss.loss_grad.at(x, y) = inv_N * 2 * pix_dif;
    }
  }
  loss.L2_loss = accum * inv_N;
}

void compute_loss(const Image &render, const Image &gt, LossData &loss) {
  L2(render, gt, loss);
}
float laplacian_smoothing(const std::vector<std::vector<int>> &adjacency,
                          const Mesh &m, std::vector<Vec3F> &grad,
                          float lambda) {
  const auto &V = m.vertices();
  size_t n = V.size();
  float scale = lambda * 2.0f / n;
  double sum = 0;

  std::vector<Vec3F> delta(n);
  for (size_t i = 0; i < n; i++) {
    Vec3F avg{0, 0, 0};
    for (int j : adjacency[i])
      avg += V[j];
    delta[i] = V[i] - avg * (1.0f / adjacency[i].size());
    sum += delta[i].dot(delta[i]);
  }

  for (size_t k = 0; k < n; k++) {
    Vec3F g = delta[k];
    for (int i : adjacency[k])
      g = g - delta[i] * (1.0f / adjacency[i].size());
    grad[k] += g * scale;
  }

  return (float)(lambda * sum / n);
}