cmake_minimum_required(VERSION 2.6)
project(OTL)

add_library(OTL src/libOTL/otl.cpp)

include_directories(src/libotl)

add_executable(otldemo src/main/otldemo.cpp)

target_link_libraries(otldemo OTL)

