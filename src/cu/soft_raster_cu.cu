#include "soft_raster_cu.h"

__constant__ CameraCUDA device_cams[MAX_CAMS]; // we use 41 but just in case

void SoftRasterizerCUDA::add_cameras(const std::vector<CameraCUDA> &cams) {
  cudaMemcpyToSymbol(device_cams, cams.data(),
                     cams.size() * sizeof(CameraCUDA));
}