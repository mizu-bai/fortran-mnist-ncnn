cmake_minimum_required(VERSION 3.13)

project(fortran_mnist_ncnn LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to gfortran. CMake otherwise picks whatever it finds first, which on
# Homebrew macOS is flang. Override with -DCMAKE_Fortran_COMPILER=...
if (NOT CMAKE_Fortran_COMPILER)
    find_program(_gfortran NAMES gfortran)
    if (_gfortran)
        set(CMAKE_Fortran_COMPILER "${_gfortran}")
    endif ()
endif ()

# ncnn is built with its own OpenMP, and with clang's libc++ on macOS. The old
# -lstdc++ is gone: that is a link flag rather than a compile flag, and libc++
# is the correct C++ standard library on Apple platforms.
if (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif ()

find_package(OpenMP REQUIRED)

# Locate ncnn. Both layouts are supported because the sources use quoted
# includes:
#
#   installed:  <prefix>/include/ncnn/net.h   +  libncnn
#   build tree: <src>/net.h                   +  <build>/src/platform.h
#                                           +  libncnn.a
#
# Point cmake at either with -Dncnn_DIR=... or the NCNN_DIR env variable.
set(_ncnn_hints HINTS ${ncnn_DIR} $ENV{NCNN_DIR})

# An explicit -Dncnn_LIBRARY= / -Dncnn_INCLUDE_DIR= wins over any search.
find_path(ncnn_INSTALL_INC NAMES ncnn/net.h ${_ncnn_hints} PATH_SUFFIXES include)
find_path(ncnn_SRC_INC     NAMES net.h       ${_ncnn_hints} PATH_SUFFIXES src)
find_path(ncnn_GEN_INC     NAMES platform.h  ${_ncnn_hints}
          PATH_SUFFIXES src build/src build-def/src)
find_library(ncnn_LIBRARY NAMES ncnn ${_ncnn_hints}
             PATH_SUFFIXES lib lib64 src build/src build-def/src)

if (ncnn_LIBRARY)
    set(ncnn_FOUND TRUE)
endif ()
if (NOT ncnn_LIBRARY)
    message(FATAL_ERROR
        "ncnn library not found. Point cmake at it with -Dncnn_DIR=/path/to/ncnn "
        "(install prefix or build tree), or name the archive directly with "
        "-Dncnn_LIBRARY=/path/to/libncnn.a.")
endif ()

if (NOT (ncnn_INSTALL_INC OR (ncnn_SRC_INC AND ncnn_GEN_INC)))
    message(FATAL_ERROR
        "ncnn headers not found. An install needs include/ncnn/net.h; a build "
        "tree needs both src/net.h and the generated src/platform.h.")
endif ()

message(STATUS "ncnn library: ${ncnn_LIBRARY}")

file(GLOB SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/*.f90
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_executable(mnist ${SRC_FILES})

# only pass the directories that were actually found
set(_ncnn_incs ${PROJECT_SOURCE_DIR}/src)
foreach (_d ${ncnn_INSTALL_INC} ${ncnn_SRC_INC} ${ncnn_GEN_INC})
    if (_d)
        list(APPEND _ncnn_incs ${_d})
    endif ()
endforeach ()

target_include_directories(mnist PRIVATE ${_ncnn_incs})
target_link_libraries(mnist PRIVATE ${ncnn_LIBRARY} OpenMP::OpenMP_CXX)

if (APPLE)
    target_link_libraries(mnist PRIVATE "-lc++")
endif ()
