cmake_minimum_required(VERSION 2.6)
project(OTL)

# OPTIONS
option (BUILD_DOCS 
        "Create Doxygen Documentation" OFF)
option (BUILD_PYTHON_BINDINGS
		"Create Python SWIG Bindings" OFF)


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
   "-Wall -Wextra -fPIC"
   )
endif()

#Add cmake modules to cmake path
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMakeResources")


set(OTL_SOURCES
    src/libOTL/otl.h
    src/libOTL/otl_exception.h
    src/libOTL/otl_helpers.h
    src/libOTL/otl_helpers.cpp
    src/libOTL/otl_kernel.h
    src/libOTL/otl_kernel_factory.h
    src/libOTL/otl_kernel_gaussian.h
    src/libOTL/otl_kernel_gaussian.cpp
    src/libOTL/otl_kernel_recursive_gaussian.h
    src/libOTL/otl_kernel_recursive_gaussian.cpp
    src/libOTL/otl_kernel_recursive_equality_gaussian.h
    src/libOTL/otl_kernel_recursive_equality_gaussian.cpp
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
    src/libOTL/otl_storkgp.h
    src/libOTL/otl_storkgp.cpp
    LICENSE.txt
)

add_library(OTL ${OTL_SOURCES})

include_directories(src/libOTL)

add_executable(oesgp_sin src/examples/oesgp_sin.cpp)

target_link_libraries(oesgp_sin OTL)

add_executable(storkgp_sin src/examples/storkgp_sin.cpp)

target_link_libraries(storkgp_sin OTL)


#DOXYGEN
if (BUILD_DOCS)
	set( PROJECT_VERSION ${0.1} )
	set( DOXYFILE_OUTPUT_DIR "${PROJECT_SOURCE_DIR}/doc" )
	set( DOXYFILE_LATEX OFF )
	set( DOXYFILE_IN "${PROJECT_SOURCE_DIR}/Doxyfile.in")
	include(UseDoxygen OPTIONAL)
endif (BUILD_DOCS)

#SWIG Bindings
if (BUILD_PYTHON_BINDINGS)
	FIND_PACKAGE(SWIG REQUIRED)
	INCLUDE(${SWIG_USE_FILE})

	FIND_PACKAGE(PythonLibs REQUIRED)
	set(INCLUDEDIRS ${PYTHON_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} src/libOTL)
	set(PYTHON_LIBRARIES)
	INCLUDE_DIRECTORIES(${INCLUDEDIRS})
	#INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
	#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
	#INCLUDE_DIRECTORIES(src/libOTL)

	SET(CMAKE_SWIG_FLAGS "-classic")

	#STORK-GP Python Bindings
	SET_SOURCE_FILES_PROPERTIES(otl_storkgp.i PROPERTIES CPLUSPLUS ON)
	SET_SOURCE_FILES_PROPERTIES(otl_storkgp.i PROPERTIES SWIG_FLAGS "-includeall")
	SWIG_ADD_MODULE(otl_storkgp python otl_storkgp.i src/libOTL/otl_storkgp.cpp)
	SWIG_LINK_LIBRARIES(otl_storkgp ${PYTHON_LIBRARIES} OTL)

	#OESGP Python Bindings
	SET_SOURCE_FILES_PROPERTIES(otl_oesgp.i PROPERTIES CPLUSPLUS ON)
	SET_SOURCE_FILES_PROPERTIES(otl_oesgp.i PROPERTIES SWIG_FLAGS "-includeall")
	SWIG_ADD_MODULE(otl_oesgp python otl_oesgp.i src/libOTL/otl_oesgp.cpp)
	SWIG_LINK_LIBRARIES(otl_oesgp ${PYTHON_LIBRARIES} OTL)
endif (BUILD_PYTHON_BINDINGS)
