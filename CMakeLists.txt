cmake_minimum_required(VERSION 3.20)
project(QuadKernelStreaming VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)

# Optimization flags
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -ffast-math")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto")

# SIMD detection
if(MSVC)
    add_compile_options(/arch:AVX2)
else()
    add_compile_options(-mavx2 -mfma)
endif()

# Global include directories
include_directories(
    quad_kernel_bridge/include
)

# Add GNAT runtime library path (from Alire toolchain - adalib contains libgnat/libgnarl)
link_directories("C:/Users/Alfred/AppData/Local/alire/cache/toolchains/gnat_native_15.2.1_346e2e00/lib/gcc/x86_64-w64-mingw32/15.2.0/adalib")

# Subdirectories
add_subdirectory(quad_kernel_bridge)
add_subdirectory(quad_kernel_c_video)
add_subdirectory(quad_kernel_cpp_audio)
add_subdirectory(quad_kernel_ada_math)

# Import Rust Static Library (built with x86_64-pc-windows-gnu target)
add_library(quad_kernel_rust_wasm STATIC IMPORTED)
set_target_properties(quad_kernel_rust_wasm PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/quad_kernel_rust_wasm/target/x86_64-pc-windows-gnu/release/libquad_kernel_rust_wasm.a"
)

# Main executable to test the full pipeline
add_executable(quad_kernel_system
    src/main_system.c
)

# Link everything together
target_link_libraries(quad_kernel_system
    quad_kernel_bridge
    video_kernel_adapter
    audio_kernel_adapter
    "${CMAKE_CURRENT_SOURCE_DIR}/quad_kernel_ada_math/lib/libquad_math.a"
    quad_kernel_rust_wasm
)

# Platform specific system libraries required by Rust/MinGW and Ada/GNAT
if(WIN32)
    target_link_libraries(quad_kernel_system
        ws2_32
        userenv
        bcrypt
        ntdll
        gnat    # Ada runtime
        gnarl   # Ada tasking runtime
    )
endif()
