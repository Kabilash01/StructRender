cmake_minimum_required(VERSION 3.20)
project(structrender_runtime LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# TensorRT: public headers fetched at the exact installed version (11.2.1) into
# third_party/; the runtime library comes from the pip wheel already used by the
# Python training/export stack, so C++ and Python run the SAME TensorRT build.
set(TRT_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorrt-oss/include" CACHE PATH "")
set(TRT_LIBDIR "" CACHE PATH "directory containing libnvinfer.so.11")
find_package(CUDAToolkit REQUIRED)
find_library(NVINFER NAMES nvinfer libnvinfer.so.11 PATHS ${TRT_LIBDIR} NO_DEFAULT_PATH REQUIRED)

add_library(structrender_rt src/engine.cpp)
target_include_directories(structrender_rt PUBLIC include ${TRT_INCLUDE})
target_link_libraries(structrender_rt PUBLIC ${NVINFER} CUDA::cudart)
set_target_properties(structrender_rt PROPERTIES BUILD_RPATH "${TRT_LIBDIR}")

add_executable(srt_run tools/srt_run.cpp)
target_link_libraries(srt_run PRIVATE structrender_rt)
set_target_properties(srt_run PROPERTIES BUILD_RPATH "${TRT_LIBDIR}")
