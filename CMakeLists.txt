cmake_minimum_required(VERSION 2.6)
project(OTL)

set(SOURCES src/libOTL/otl.h
    src/libOTL/otl.cpp
    src/libOTL/otl_exception.h
    src/libOTL/otl_kernels.h
    src/libOTL/otl_learning_algs.h
    src/libOTL/otl_rls.h
    src/libOTL/otl_aug_feature.h
    src/libOTL/otl_reservoir.h
    src/libOTL/otl_window.h
    src/libOTL/otl_helpers.h
)
add_library(OTL ${SOURCES})

#src/libOTL/otl.cpp src/libOTL/*.h)

include_directories(src/libotl)

add_executable(otldemo src/main/otldemo.cpp)

target_link_libraries(otldemo OTL)

