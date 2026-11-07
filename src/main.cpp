#include "geometry.h"
#include "image.h"
#include "loss.h"
#include "mesh.h"
#include "persp_camera.h"
#include "soft_raster.h"
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>

int main() {

  SoftRasterizerCPU raster;
  int n_views = 41;
  int n_iters = 1;
  // 10000;
  std::string dataset = "cat";
  float scale;
  std::string path = "../data/" + dataset + "/";
  std::filesystem::create_directories(path + "train_views");
  std::filesystem::create_directories(path + "render_views");
  std::filesystem::create_directories(path + "opt_views");

  bool generate = false;

  Mesh gt_mesh;
  load_obj(path + dataset + ".obj", gt_mesh);
  Vec3F center = gt_mesh.get_center_bbox(scale);

  Mesh opt_mesh;
  create_sphere(opt_mesh, center, 0.7 * scale, NUM_FACES_START);

  LossData loss;
  std::vector<int> order(n_views);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(42);

  std::vector<Image> gt_views;

  for (int i = 0; i < n_views; i++) {
    // create cam
    PerspectiveCamera cam(WIDTH, HEIGHT,
                          Mat3{400, 0, WIDTH / 2, 0, 400, HEIGHT / 2, 0, 0, 1});

    cam.set_pos({center.x(), center.y(), center.z()});
    cam.update_pos(i, n_views, scale * 1.2);
    cam.look_at(center);

    raster.add_camera(cam);

    Image gt(WIDTH, HEIGHT);
    gt.read_from_file(path + "train_views/out_" + std::to_string(i));
    gt_views.push_back(gt);
  }

  for (int iter = 0; iter < n_iters; iter++) {

    std::shuffle(order.begin(), order.end(), rng);
    double batch_loss = 0;

    for (int b = 0; b < BATCH_SIZE; b++) {
      int idx = order[b];
      raster.forward(opt_mesh, idx);
      compute_loss(raster.get_image(), gt_views[idx], loss);
      batch_loss += loss.L2_loss;
    }

    // optimizer step (once per batch)

    if (iter % 20 == 0) {
      printf("iter %d  loss %.6f\n", iter, batch_loss);
      for (int idx : {0, 7, 14, 26}) {
        Image shaded(WIDTH, HEIGHT);
        raster.render_shaded(opt_mesh, idx, shaded);
        shaded.write_to_file(path + "opt_views/shaded_" + std::to_string(idx) +
                             "_it" + std::to_string(iter) + ".pgm");
      }
    }
  }
}