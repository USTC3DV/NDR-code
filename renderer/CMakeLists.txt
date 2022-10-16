cmake_minimum_required(VERSION 3.0.0)
project(RenderUtils VERSION 1.0.0)

find_package(pybind11 REQUIRED)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
message("LIB Path: ${LIBRARY_OUTPUT_PATH}")

pybind11_add_module(${PROJECT_NAME} render_func.cpp)
# add_library(RenderUtils render_func.cpp)
target_link_libraries(RenderUtils PUBLIC pybind11::module)