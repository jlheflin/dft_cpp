cmake_minimum_required(VERSION 3.10)
project(dft_cpp LANGUAGES CXX C Fortran)
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_ORIGIN "$ORIGIN/../lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(BUILD_SHARED_LIBS ON)


find_package(spdlog CONFIG)
if(NOT spdlog_FOUND)
  FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.17.0
  )
  FetchContent_MakeAvailable(spdlog)
endif()

find_package(Eigen3 CONFIG NO_MODULE)
if(NOT Eigen3_FOUND)
  FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0.1
  )
  FetchContent_MakeAvailable(Eigen3)
endif()

find_package(Libxc CONFIG)
if(NOT Libxc_FOUND)
  FetchContent_Declare(
    libxc
    GIT_REPOSITORY https://gitlab.com/libxc/libxc.git
    GIT_TAG 7.0.0
  )
  FetchContent_MakeAvailable(libxc)
endif()

find_package(libint2 CONFIG)
if(NOT libint2_FOUND)
  FetchContent_Declare(
    libint2
    URL https://github.com/evaleev/libint/releases/download/v2.13.1/libint-2.13.1.tgz
    URL_HASH SHA256=029e1347774bbf740e8ef9118d736413b52c023b85a5f76b61ce6fbf1ab8e43b
  )
  FetchContent_MakeAvailable(libint2)
endif()

find_package(OpenMP REQUIRED COMPONENTS C CXX Fortran)
find_package(OpenBLAS REQUIRED)
find_package(TBB REQUIRED)
find_package(fmt REQUIRED)

file(DOWNLOAD
  "https://people.sc.fsu.edu/~jburkardt/cpp_src/sphere_lebedev_rule/sphere_lebedev_rule.cpp"
  "${CMAKE_BINARY_DIR}/sphere_lebedev_rule.cpp"
)
file(DOWNLOAD
  "https://people.sc.fsu.edu/~jburkardt/cpp_src/sphere_lebedev_rule/sphere_lebedev_rule.hpp"
  "${CMAKE_BINARY_DIR}/sphere_lebedev_rule.hpp"
)
add_library(lebedev SHARED
  ${CMAKE_BINARY_DIR}/sphere_lebedev_rule.cpp
)
target_include_directories(lebedev PUBLIC
  ${CMAKE_BINARY_DIR}
)

add_executable(dft_code src/main.cpp)
target_include_directories(dft_code PRIVATE
  ${libxc_SOURCE_DIR}/src
  ${libxc_BINARY_DIR}
)
target_link_libraries(dft_code PRIVATE
  Libint2::int2
  Eigen3::Eigen
  xc
  lebedev
  OpenBLAS::OpenBLAS
  OpenMP::OpenMP_CXX
  fmt::fmt
  TBB::tbb
  spdlog::spdlog
)
install(TARGETS dft_code lebedev
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
