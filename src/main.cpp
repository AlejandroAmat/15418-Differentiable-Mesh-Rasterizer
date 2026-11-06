#include "geometry.h"
#include "image.h"
#include "mesh.h"
#include "persp_camera.h"
#include "soft_raster.h"
#include <iostream>

int main() {

  // For nown lets just try to create a mesh and soft rasterize it.
  Mesh cube_mesh;
  create_cube(cube_mesh, {0, 0, 5}, 2);

  PerspectiveCamera cam(1200, 1200, Mat3{1200, 0, 600, 0, 1200, 600, 0, 0, 1});

  SoftRasterizerCPU raster;
  raster.add_camera(cam);

  Image render(1200, 1200);
  raster.forward(cam, cube_mesh, render);

  render.write_to_file("out.pgm");
}