cmake_minimum_required(VERSION 3.18)
project(perceptnet_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PERCEPTNET_BUILD_PYTHON "Build the pybind11 extension module" ON)
option(PERCEPTNET_BUILD_BENCH "Build the standalone C++ benchmark" ON)
option(PERCEPTNET_WITH_OPENCV "Build OpenCV preprocessing kernels" ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -O3)
endif()

if(PERCEPTNET_WITH_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core imgproc)
  if(NOT OpenCV_FOUND)
    message(WARNING "OpenCV not found: building without C++ preprocessing kernels")
    set(PERCEPTNET_WITH_OPENCV OFF)
  endif()
endif()

set(CORE_SOURCES src/nms.cpp)
if(PERCEPTNET_WITH_OPENCV)
  list(APPEND CORE_SOURCES src/preprocess.cpp)
endif()

add_library(perceptnet_core STATIC ${CORE_SOURCES})
target_include_directories(perceptnet_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(PERCEPTNET_WITH_OPENCV)
  target_compile_definitions(perceptnet_core PUBLIC PERCEPTNET_WITH_OPENCV)
  target_include_directories(perceptnet_core PUBLIC ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(perceptnet_core PUBLIC ${OpenCV_LIBS})
endif()

if(PERCEPTNET_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  # Locate pybind11 installed via pip if it is not already on CMAKE_PREFIX_PATH.
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(PYBIND11_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
  endif()
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_perceptnet_cpp src/bindings.cpp)
  target_link_libraries(_perceptnet_cpp PRIVATE perceptnet_core)
  # Drop the built module straight into the Python package so `import perceptnet` finds it.
  set_target_properties(_perceptnet_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../perceptnet/ops
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/../perceptnet/ops)
endif()

if(PERCEPTNET_BUILD_BENCH)
  add_executable(bench_ops bench/bench_ops.cpp)
  target_link_libraries(bench_ops PRIVATE perceptnet_core)
endif()
