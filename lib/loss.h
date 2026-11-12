#pragma once
#include "image.h"
#include "mesh.h"
#include "utils.h"
#include <cmath>

struct LossData {
  float L2_loss;
  float L2_rgb;
  Image loss_grad{WIDTH, HEIGHT, 4};
};

void L2(const Image &render, const Image &gt, LossData &loss, float w_rgb);

float laplacian_smoothing(const std::vector<std::vector<int>> &adjacency,
                          const Mesh &m, std::vector<Vec3F> &grad,
                          float lambda);

void compute_loss(const Image &render, const Image &gt, LossData &loss,
                  float w_rgb);