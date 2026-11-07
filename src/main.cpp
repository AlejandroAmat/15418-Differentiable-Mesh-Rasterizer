#include "geometry.h"
#include "image.h"
#include "mesh.h"
#include "persp_camera.h"
#include "soft_raster.h"
#include <filesystem>
#include <iostream>

int main() {

  // For nown lets just try to create a mesh and soft rasterize it.
  // Mesh cube_mesh;
  // create_cube(cube_mesh, {0, -2, 10}, 2);ç

  SoftRasterizerCPU raster;
  int n_views = 41;
  int n_iters = 1;
  // 10000;
  std::string dataset = "cat";
  float scale;
  std::string path = "../data/" + dataset + "/";
  // std::filesystem::create_directories(path + "train_views");
  // std::filesystem::create_directories(path + "render_views");
  std::filesystem::create_directories(path + "opt_views");

  Mesh gt_mesh;
  load_obj(path + dataset + ".obj", gt_mesh);
  Vec3F center = gt_mesh.get_center_bbox(scale);

  Mesh opt_mesh;
  create_sphere(opt_mesh, center, 0.7 * scale, NUM_FACES_START);

  std::vector<Image> gt_views;
  // generate data
  for (int i = 0; i < n_views; i++) {
    // create cam
    PerspectiveCamera cam(
        WIDTH, HEIGHT, Mat3{1200, 0, WIDTH / 2, 0, 1200, HEIGHT / 2, 0, 0, 1});

    cam.set_pos({center.x(), center.y(), center.z()});
    cam.update_pos(i, n_views, scale * 1.2);
    cam.look_at(center);

    raster.add_camera(cam);

    Image render(WIDTH, HEIGHT);
    raster.add_view(std::move(render));

    Image gt(WIDTH, HEIGHT);
    gt.read_from_file(path + "train_views/out_" + std::to_string(i));
    gt_views.push_back(gt);
  }

  for (int iter = 0; iter < n_iters; iter++) {
    for (int idx = 0; idx < n_views; idx++) {

      // fw
      raster.forward(opt_mesh, idx);
      Image &render = raster.get_image(idx);
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