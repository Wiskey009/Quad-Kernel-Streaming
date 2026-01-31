cmake_minimum_required(VERSION 3.10)
project(quad_kernel_cpp_audio VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
if(MSVC)
    add_compile_options(/W4 /O2 /arch:AVX2)
else()
    add_compile_options(-Wall -Wextra -O3 -mavx2 -march=native)
endif()

include_directories(include)

# Find necessary libraries (optional for now, can be added as needed)
# find_package(Threads REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h" "include/*.hpp")

add_library(audio_kernel_adapter STATIC ${SOURCES} ${HEADERS})
target_link_libraries(audio_kernel_adapter quad_kernel_bridge)

# Add an example test executable
add_executable(test_audio src/test_audio.cpp)
target_link_libraries(test_audio audio_kernel_adapter)
