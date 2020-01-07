cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)
project(slmc_final)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set (CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda/")
set(CMAKE_PREFIX_PATH /usr/local/include/libtorch/share)

find_package(Torch REQUIRED)

add_executable(slmc_final main.cpp dataload.h neuralnetwork.h slmc.h)
target_link_libraries(slmc_final "${TORCH_LIBRARIES}" armadillo)

set_property(TARGET slmc_final PROPERTY CXX_STANDARD 14)