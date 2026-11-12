#include "loss.h"

void L2(const Image &render, const Image &gt, LossData &loss, float w_rgb) {
  double acc_a = 0, acc_rgb = 0;
  float inv_N = 1.0f / (WIDTH * HEIGHT * BATCH_SIZE);

  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      // silhouette (alpha)
      float da = render.at(x, y, 3) - gt.at(x, y, 3);
      acc_a += da * da;
      loss.loss_grad.at(x, y, 3) = inv_N * 2 * da;

      // color
      for (int c = 0; c < 3; ++c) {
        float dc = render.at(x, y, c) - gt.at(x, y, c);
        acc_rgb += dc * dc;
        loss.loss_grad.at(x, y, c) = w_rgb * inv_N * 2 * dc;
      }
    }
  }
  loss.L2_loss = acc_a * inv_N;
  loss.L2_rgb = w_rgb * acc_rgb * inv_N;
}

void compute_loss(const Image &render, const Image &gt, LossData &loss,
                  float w_rgb) {
  L2(render, gt, loss, w_rgb);
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