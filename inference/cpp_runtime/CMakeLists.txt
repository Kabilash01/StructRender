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

# --- GDExtension: StructRenderer for Godot (M10) -------------------------------
set(GODOT_CPP "${CMAKE_CURRENT_SOURCE_DIR}/third_party/godot-cpp")
if(EXISTS "${GODOT_CPP}/bin/libgodot-cpp.linux.template_debug.x86_64.a")
  add_library(structrender_gd SHARED gdextension/register_types.cpp gdextension/structrender_node.cpp)
  target_include_directories(structrender_gd PRIVATE gdextension
      ${GODOT_CPP}/include ${GODOT_CPP}/gen/include ${GODOT_CPP}/gdextension)
  target_compile_definitions(structrender_gd PRIVATE DEBUG_ENABLED)
  target_link_libraries(structrender_gd PRIVATE structrender_rt
      ${GODOT_CPP}/bin/libgodot-cpp.linux.template_debug.x86_64.a)
  set_target_properties(structrender_gd PROPERTIES
      OUTPUT_NAME structrender PREFIX "lib"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../game_project/bin"
      BUILD_RPATH "${TRT_LIBDIR}" INSTALL_RPATH "${TRT_LIBDIR}")
endif()
