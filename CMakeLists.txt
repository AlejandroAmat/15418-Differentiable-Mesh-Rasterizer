cmake_minimum_required(VERSION 3.18)
project(diff_rast_418 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(diff_rast_418
  src/main.cpp
  src/core/soft_raster.cpp
  src/scene/persp_camera.cpp
  src/scene/mesh.cpp
  src/helpers/geometry.cpp
)

target_include_directories(diff_rast_418 PRIVATE lib)