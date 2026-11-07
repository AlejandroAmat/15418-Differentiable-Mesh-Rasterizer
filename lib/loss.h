#pragma once
#include "image.h"
#include "utils.h"
#include <cmath>

struct LossData {
  float L2_loss;
  Image loss_grad{WIDTH, HEIGHT};
};

void L2(const Image &render, const Image &gt, LossData &loss);

void compute_loss(const Image &render, const Image &gt, LossData &loss);