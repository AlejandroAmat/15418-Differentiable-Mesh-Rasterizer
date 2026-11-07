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