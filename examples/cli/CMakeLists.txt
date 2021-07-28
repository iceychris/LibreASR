cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cli)

find_package(Torch REQUIRED)

add_executable(libreasr libreasr.cpp)
target_link_libraries(libreasr "${TORCH_LIBRARIES}")
set_property(TARGET libreasr PROPERTY CXX_STANDARD 14)
