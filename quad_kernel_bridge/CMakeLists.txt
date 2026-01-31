add_library(quad_kernel_bridge STATIC
    src/kernel_bridge.c
)

target_include_directories(quad_kernel_bridge PUBLIC include)
