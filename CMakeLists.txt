cmake_minimum_required(VERSION 2.6)
project(OTL)

#===============================================================================
# COMPILER SETTINGS (Warnings and visibility)
#===============================================================================
#Taken from Stack Overflow (question: 2368811, user: mloskot)
if(MSVC)
 # Force to always compile with W4
 if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
   string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
 else()
   list(APPEND CMAKE_CXX_FLAGS "/W4")
 endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
 # Update if necessary
 list(APPEND CMAKE_CXX_FLAGS
   "-Wall -Wextra"
   )
endif()


set(SOURCES src/libOTL/otl.h
    src/libOTL/otl_exception.h
    src/libOTL/otl_helpers.h
    src/libOTL/otl_helpers.cpp
    src/libOTL/otl_kernel.h
    src/libOTL/otl_kernel_factory.h
    src/libOTL/otl_kernel_gaussian.h
    src/libOTL/otl_kernel_gaussian.cpp
    src/libOTL/otl_kernel_recursive_gaussian.h
    src/libOTL/otl_kernel_recursive_gaussian.cpp
    src/libOTL/otl_learning_algs.h
    src/libOTL/otl_rls.h
    src/libOTL/otl_rls.cpp
    src/libOTL/otl_aug_state.h
    src/libOTL/otl_reservoir.h
    src/libOTL/otl_reservoir.cpp
    src/libOTL/otl_window.h
    src/libOTL/otl_window.cpp
    src/libOTL/otl_sogp.h
    src/libOTL/otl_sogp.cpp
    src/libOTL/otl_oesgp.h
    src/libOTL/otl_oesgp.cpp
    LICENSE.txt
)

add_library(OTL ${SOURCES})

include_directories(src/libotl)

add_executable(otldemo src/main/otldemo.cpp)

target_link_libraries(otldemo OTL)




