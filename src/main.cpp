#include "geometry.h"
#include "image.h"
#include "loss.h"
#include "mesh.h"
#include "optimizer.h"
#include "persp_camera.h"
#include "soft_raster.h"
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>

#include <chrono>
#include <cstdio>

int main(int argc, char **argv) {
  float lambda = (argc > 1) ? std::stof(argv[1]) : 480.0f;
  std::string run = (argc > 1) ? argv[1] : "480";

  SoftRasterizerCPU raster;
  int n_views = 41;
  int n_iters = 800;
  // 10000;

  float scale;

  std::string dataset = "dragon";
  std::string path = "../data/" + dataset + "/";

  std::filesystem::create_directories(path + "train_views");
  std::filesystem::create_directories(path + "render_views");
  std::filesystem::create_directories(path + "opt_views");
  std::string out_dir = path + "opt_views/lambda_" + run + "/";
  std::filesystem::create_directories(out_dir);

  bool generate = false;

  Mesh gt_mesh;
  load_obj(path + dataset + ".obj", gt_mesh);
  Vec3F c = gt_mesh.get_center_bbox(scale);
  for (Vec3F &v : gt_mesh.vertices())
    v = (v - c) * (1.0f / scale);

  Vec3F center{0, 0, 0};
  scale = 1.0f;

  Mesh opt_mesh;
  create_sphere(opt_mesh, center, 0.8 * scale, NUM_FACES_START);
  std::vector<std::vector<int>> adjacency = build_adjacency(opt_mesh);

  LossData loss;
  std::vector<int> order(n_views);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(42);

  float lr = 0.0060f;
  Adam optim(lr);
  std::vector<Vec3F> grad(opt_mesh.vertices().size());

  std::vector<Image> gt_views;

  int last_subdiv = 0;

  for (int i = 0; i < n_views; i++) {
    // create cam
    PerspectiveCamera cam(WIDTH, HEIGHT,
                          Mat3{400, 0, WIDTH / 2, 0, 400, HEIGHT / 2, 0, 0, 1});

    cam.set_pos({center.x(), center.y(), center.z()});
    cam.update_pos(i, n_views, scale * 1.2);
    cam.look_at(center);

    raster.add_camera(cam);

    if (generate) {
      raster.forward(gt_mesh, i);
      raster.get_image().write_to_file(path + "train_views/out_" +
                                       std::to_string(i) + ".pgm");
    }

    Image gt(WIDTH, HEIGHT);
    gt.read_from_file(path + "train_views/out_" + std::to_string(i) + ".pgm");
    gt_views.push_back(std::move(gt));
  }

  for (int iter = 0; iter < n_iters; iter++) {

    float t = std::min(iter / 200.0f, 1.0f);
    raster.set_sigma(12.0f * std::pow(0.1f / 12.0f, t));

    if (iter == 200 || iter == 400) {
      opt_mesh.subdivide_midpoint();
      adjacency = build_adjacency(opt_mesh);
      lambda *= 7;
      lr *= 0.8f;
      optim.set_lr(lr);
      last_subdiv = iter;
    }

    int since = iter - last_subdiv;
    float warm = std::min(1.0f, 0.1f + since / 20.0f);
    optim.set_lr(lr * warm);

    std::shuffle(order.begin(), order.end(), rng);
    double batch_loss = 0;
    grad.assign(opt_mesh.vertices().size(), Vec3F{0, 0, 0});

    for (int b = 0; b < BATCH_SIZE; b++) {

      int idx = order[b];
      raster.forward(opt_mesh, idx);

      compute_loss(raster.get_image(), gt_views[idx], loss);
      batch_loss += loss.L2_loss;

      raster.backward(loss, opt_mesh, idx, grad);
    }

    float lapl_loss = laplacian_smoothing(adjacency, opt_mesh, grad, lambda);

    optim.step(opt_mesh.vertices(), grad);
    // subdivision...Maybe

    if (iter % 5 == 0) {
      printf("iter %d  loss %.6f\n", iter, batch_loss);
      printf("iter %d  loss_lapl %.6f\n", iter, lapl_loss);
      for (int idx : {0, 7, 14, 26}) {

        Image shaded(WIDTH, HEIGHT);
        raster.render_shaded(opt_mesh, idx, shaded);
        shaded.write_to_file(out_dir + "shaded_" + std::to_string(idx) + "_f" +
                             std::to_string(iter / 5) + ".pgm");
      }
    }
  }
}