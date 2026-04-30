cmake_minimum_required(VERSION 3.16)
project(matt CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(matt
    src/storage.cpp
    src/tensor.cpp
    src/ops.cpp
)
target_include_directories(matt PUBLIC include)

add_executable(tests tests/test_tensor.cpp)
target_link_libraries(tests matt)

add_custom_target(format
    COMMAND find ${CMAKE_SOURCE_DIR} -type f -name "*.cpp" -exec clang-format -i {} +
    COMMAND find ${CMAKE_SOURCE_DIR} -type f -name "*.h"   -exec clang-format -i {} +
)