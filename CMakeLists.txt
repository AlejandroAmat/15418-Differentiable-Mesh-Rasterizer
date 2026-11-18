cmake_minimum_required(VERSION 3.18)
set(CMAKE_C_COMPILER   /usr/bin/gcc-11)
set(CMAKE_CXX_COMPILER /usr/bin/g++-11)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-11)
project(diff_rast_418 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(diff_rast_418
  src/main.cpp
  src/core/soft_raster.cpp
  src/scene/persp_camera.cpp
  src/scene/mesh.cpp
  src/helpers/geometry.cpp
  src/core/image.cpp
  src/optim/loss.cpp
  src/optim/optimizer.cpp
  src/core/texture.cpp
  src/cu/soft_raster_cu.cu
)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(diff_rast_418 PRIVATE CUDA::cudart)

target_include_directories(diff_rast_418 PRIVATE lib src/cu)
