cmake_minimum_required(VERSION 2.8.9)

project(eigensolver CXX)
enable_language(C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(ARPACK arpack)
pkg_check_modules(LAPACK lapack)
include_directories(${ARPACK_INCLUDE_DIRS})

add_subdirectory(googletest)
include_directories(${CMAKE_SOURCE_DIR}/googletest/include)
link_directories(${googletest_BINARY_DIR}/src)
enable_testing()
set(testdata_DIR "${PROJECT_SOURCE_DIR}/test-data")
include_directories(${CMAKE_SOURCE_DIR}/googletest/googletest/include)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(FindSuiteSparse)
include_directories(${SUITESPARSE_INCLUDE_DIRS})

set(CXX_WARN_FLAGS "-W -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS "-g -std=c++11 ${CXX_WARN_FLAGS} ${OpenMP_CXX_FLAGS} ${ARPACK_CFLAGS}")

add_library(eigensolver eigen-solver.cpp linear-algebra.cpp linear-solver-spqr.cpp)

add_executable(test-cholmod test-cholmod.cpp)
target_link_libraries(test-cholmod ${SUITESPARSE_LIBRARIES} gtest)

add_executable(test-eigen-solver test-eigen-solver.cpp)
target_link_libraries(test-eigen-solver eigensolver ${ARPACK_LDFLAGS} ${SUITESPARSE_LIBRARIES})

add_executable(test-linear-solver test-linear-solver.cpp)
target_link_libraries(test-linear-solver eigensolver ${ARPACK_LDFLAGS} ${SUITESPARSE_LIBRARIES} ${BLAS_LIBRARIES} gtest)

add_test(cholmod test-cholmod)
add_test(eigen-solver-1 test-eigen-solver ${testdata_DIR}/A.mtx /dev/null)
add_test(eigen-solver-2 test-eigen-solver ${testdata_DIR}/K.mtx /dev/null)
add_test(eigen-solver-3 test-eigen-solver ${testdata_DIR}/K-ref.mtx /dev/null)
add_test(linear-solver test-linear-solver)
