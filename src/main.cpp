#include "device_buffer.h"
#include "geometry.h"
#include "image.h"
#include "loss.h"
#include "mesh.h"
#include "optimizer.h"
#include "optimizer_cu.h"
#include "persp_camera.h"
#include "soft_raster.h"
#include "soft_raster_cu.h"
#include "structs_cu.h"
#include "texture.h"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>

int main(int argc, char **argv) {
  float lambda = (argc > 1) ? std::stof(argv[1]) : 420.0f;
  std::string run = (argc > 1) ? argv[1] : "420";

  int CUDA_BATCH_SIZE = 20;

  SoftRasterizerCPU raster;
  SoftRasterizerCUDA cuda_raster{WIDTH, HEIGHT, CUDA_BATCH_SIZE};

  int n_views = 41;
  int n_iters = 600;
  // 10000;

  float scale;

  bool USE_CUDA = true;
  bool USE_CPP = false;

  std::string dataset = "demon";
  std::string path = "../data/" + dataset + "/";

  std::filesystem::create_directories(path + "train_views");
  std::filesystem::create_directories(path + "render_views");
  std::filesystem::create_directories(path + "opt_views");
  std::string out_dir = path + "opt_views/lambda_" + run + "/";
  std::filesystem::create_directories(out_dir);

  bool generate = false;

  // GT
  Mesh gt_mesh;
  load_obj(path + dataset + ".obj", gt_mesh);
  Vec3F c = gt_mesh.get_center_bbox(scale);
  for (Vec3F &v : gt_mesh.vertices())
    v = (v - c) * (1.0f / scale);

  Vec3F center{0, 0, 0};
  scale = 1.0f;
  Texture gt_tex;
  gt_tex.load(path + "material_0.png");

  //------- Optim ----------
  // :::::Mesh:::
  // host
  Mesh opt_mesh;
  create_sphere(opt_mesh, center, 0.8 * scale, NUM_FACES_START);
  compute_sphere_uvs(opt_mesh, center);
  std::vector<std::vector<int>> adjacency = build_adjacency(opt_mesh);
  std::vector<Vec3F> grad(opt_mesh.vertices().size());
  // device
  MeshCUDA opt_mesh_device;

  //:::::::::::::::
  //::::Text::::::
  // host
  Texture opt_tex;
  opt_tex.init(512, 512, 0.5f);
  Image tex_grad(opt_tex.get_w(), opt_tex.get_h(), 3);
  TextureCUDA opt_tex_device;

  // ADAM params
  float lr_tex = 0.02f;
  Adam tex_optim(lr_tex);
  float lr = 0.0060f;
  Adam optim(lr);

  AdamCUDA tex_optim_cu(lr_tex);
  AdamCUDA mesh_optim_cu(lr);

  int tex_start = 50;
  LossData loss;

  // random batches
  std::vector<int> order(n_views);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(42);

  // GT
  std::vector<Image> gt_views;

  int last_subdiv = -50;
  std::vector<CameraCUDA> device_cams;

  for (int i = 0; i < n_views; i++) {
    // create cam
    PerspectiveCamera cam(
        WIDTH, HEIGHT,
        Mat3{WIDTH, 0, WIDTH / 2, 0, HEIGHT, HEIGHT / 2, 0, 0, 1});

    cam.set_pos({center.x(), center.y(), center.z()});
    cam.update_pos(i, n_views, scale * 1.2);
    cam.look_at(center);

    raster.add_camera(cam);

    CameraCUDA device_cam;
    device_cam.from_cam(cam);
    device_cams.push_back(device_cam);

    if (generate) {
      raster.set_sigma(0.1f);
      raster.forward(gt_mesh, gt_tex, i);
      raster.get_image().write_png(path + "train_views/out_" +
                                   std::to_string(i) + ".png");
    }

    Image gt(WIDTH, HEIGHT, 4);
    gt.read_png(path + "train_views/out_" + std::to_string(i) + ".png");
    gt_views.push_back(std::move(gt));
  }

  if (USE_CUDA) {
    cuda_raster.add_cameras(device_cams);
    opt_tex_device.upload_to_device(opt_tex);
    opt_mesh_device.upload_to_device(opt_mesh);
    cuda_raster.init_allocate(n_views, (int)opt_mesh.faces().size(), gt_views,
                              opt_mesh.vertices().size());
  }

  for (int iter = 0; iter < n_iters; iter++) {

    float t = std::min(iter / 200.0f, 1.0f);
    raster.set_sigma(7.0f * std::pow(0.1f / 12.0f, t));
    cuda_raster.set_sigma(7.0f * std::pow(0.1f / 12.0f, t));

    //-------------subdivision updates--------------
    if (iter == 160 || iter == 240 || iter == 320) {
      opt_mesh.subdivide_midpoint();
      adjacency = build_adjacency(opt_mesh);
      lambda *= 2;
      lr *= 0.9;
      optim.set_lr(lr);
      last_subdiv = iter;
    }
    if (iter > 400 && iter % 30 == 0) {
      lambda *= 0.8;
    }

    int since = iter - last_subdiv;
    float warm = std::min(1.0f, 0.1f + since / 10.0f);
    optim.set_lr(lr * warm);
    // mesh_optim_cu.set_lr(lr * warm);
    //------------------------------------------------

    std::shuffle(order.begin(), order.end(), rng);
    double batch_loss = 0;
    double batch_rgb = 0;
    grad.assign(opt_mesh.vertices().size(), Vec3F{0, 0, 0});
    tex_grad.fill(0.0f);

    float w_rgb = (iter >= tex_start) ? 1.0f : 0.0f;

    //---------------------------------------------------

    int rounds = BATCH_SIZE / CUDA_BATCH_SIZE;

    opt_mesh_device.grads.set_zero();
    opt_tex_device.grads.set_zero();

    auto start_gpu = std::chrono::high_resolution_clock::now();
    for (int b = 0; b < rounds; ++b) {

      if (USE_CUDA) {
        // GPU
        cuda_raster.forward(opt_mesh_device, opt_tex_device, order,
                            b * CUDA_BATCH_SIZE);
        cuda_raster.backward(opt_mesh_device, opt_tex_device, order,
                             b * CUDA_BATCH_SIZE, w_rgb);
      }

      if (USE_CPP) {
        for (int i = 0; i < CUDA_BATCH_SIZE; ++i) {
          // cpu
          int idx = order[b * CUDA_BATCH_SIZE + i];
          raster.forward(opt_mesh, opt_tex, idx);

          compute_loss(raster.get_image(), gt_views[idx], loss, w_rgb);
          batch_loss += loss.L2_loss;
          batch_rgb += loss.L2_rgb;

          raster.backward(loss, opt_mesh, opt_tex, idx, grad, tex_grad);
        }
      }
    }

    // float lapl_loss = laplacian_smoothing(adjacency, opt_mesh, grad,
    // lambda);

    // optim.step(opt_mesh.vertices(), grad);
    // if (iter >= tex_start)
    //   tex_optim.step(&opt_tex.image().at(0, 0), &tex_grad.at(0, 0),
    //                  (size_t)opt_tex.get_w() * opt_tex.get_h() * 3);

    mesh_optim_cu.step(opt_mesh_device.verts.ptr, opt_mesh_device.grads.ptr,
                       opt_mesh_device.n_verts * 3);
    if (iter >= tex_start)
      tex_optim_cu.step(opt_tex_device.data.ptr, opt_tex_device.grads.ptr,
                        (size_t)opt_tex_device.w * opt_tex_device.h * 3);

    cudaDeviceSynchronize();
    auto end_gpu_fwd = std::chrono::high_resolution_clock::now();
    auto duration_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_gpu_fwd - start_gpu);

    std::cout << "\n GPU: " << duration_gpu.count() << std::endl;

    if (iter % 5 == 0) {
      // printf("iter %d  loss %.6f\n", iter, batch_loss);
      // printf("iter %d  loss_lapl %.6f\n", iter, lapl_loss);
      // printf("iter %d  loss_tex %.6f\n", iter, batch_rgb);
      // for (int idx : {0¿, 7, 14, 26}) {
      for (int idx : {0}) {

        Image shaded(WIDTH, HEIGHT, 4);

        opt_mesh_device.verts.get_from_device(
            (float *)opt_mesh.vertices().data());
        opt_tex_device.data.get_from_device(&opt_tex.image().at(0, 0));

        raster.render_shaded(opt_mesh, opt_tex, idx, shaded);
        // shaded.write_png(out_dir + "shaded_" + std::to_string(idx) + "_f" +
        //                  std::to_string(iter / 5) + ".png");
        shaded.write_png(out_dir + "shaded_" + std::to_string(idx) + "_f" +
                         std::to_string('final') + ".png");
      }
    }
  }
}