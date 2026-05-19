cmake_minimum_required(VERSION 3.16)
project(matt CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA
option(MATT_CUDA "Enable CUDA support" OFF)

if(MATT_CUDA)
    enable_language(CUDA) # Find nvcc, set up rules to compile .cu files.
    set(CMAKE_CUDA_STANDARD 17)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80 90) # Make nvcc generate code for SM80, SM90.
    endif()
endif()


# Library
file(GLOB_RECURSE MATT_SOURCES "src/*.cpp")

if(MATT_CUDA)
    file(GLOB_RECURSE MATT_CUDA_SOURCES "src/*.cu")
    list(APPEND MATT_SOURCES ${MATT_CUDA_SOURCES})
endif()

add_library(matt ${MATT_SOURCES})
target_include_directories(matt PUBLIC include)
set_target_properties(matt PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MATT_CUDA)
    find_package(CUDAToolkit REQUIRED) # Find CUDA toolkit to obtain targets such as cudart, cublas etc.
    target_compile_definitions(matt PUBLIC MATT_CUDA)
    set_target_properties(matt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(matt PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Google test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# Python module
pybind11_add_module(matt_python bindings/matt_bindings.cpp)
target_link_libraries(matt_python PRIVATE matt)
target_include_directories(matt_python PRIVATE include)
set_target_properties(matt_python PROPERTIES OUTPUT_NAME "matt")

# Executable
enable_testing()

file(GLOB_RECURSE TEST_SOURCES tests/*.cpp)
add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests
    matt
    GTest::gtest_main
)

add_custom_target(format
    COMMAND find ${CMAKE_SOURCE_DIR} -type f -name "*.cpp" -exec clang-format -i {} +
    COMMAND find ${CMAKE_SOURCE_DIR} -type f -name "*.h"   -exec clang-format -i {} +
)

include(GoogleTest)
gtest_discover_tests(tests)