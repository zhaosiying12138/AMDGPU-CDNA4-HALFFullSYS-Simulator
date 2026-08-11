cmake_minimum_required(VERSION 3.20)

project(TritonGemsimAMD LANGUAGES C CXX)

# This backend reuses Triton's in-tree AMD lowering. Its product-specific
# launcher is Python/ctypes over the repository-local self runtime, so it does
# not add another compiler dialect library. Triton's plugin loader still
# requires one initializer for the generated libtriton submodule.
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonGemsimAMD triton_gemsim_amd.cc)
endif()
