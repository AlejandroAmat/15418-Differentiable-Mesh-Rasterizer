#include "geometry.h"
#include "image.h"
#include "mesh.h"
#include "persp_camera.h"
#include "soft_raster.h"
#include <filesystem>
#include <iostream>

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
    for (int idx = 0; idx < n_views; idx++) {

      if (generate) {
        raster.forward(gt_mesh, idx);
        Image &render = raster.get_image();
        render.write_to_file(path + "train_views/out_" + std::to_string(idx) +
                             ".pgm");
      }
      // fw
      raster.forward(opt_mesh, idx);
      Image &render = raster.get_image();
      Image &gt_render = gt_views[idx];

      // loss

      // bw

      // optimizer

      if (iter % 20 == 0) {
        if (idx == 0 || idx == 7 || idx == 14 || idx == 26) {
          Image shaded(WIDTH, HEIGHT);
          raster.render_shaded(opt_mesh, idx, shaded);
          shaded.write_to_file(path + "opt_views/shaded_" +
                               std::to_string(idx) + ".pgm");
        }
      }
    }
  }
}