add_library(video_kernel_adapter STATIC
    video_kernel_adapter.c
    hvcal_core.c
    nvenc_manager.c
    color_converter.c
    motion_engine.c
    rate_controller.c
    transform_quant.c
    entropy_coder.c
    filter_chain.c
)

target_include_directories(video_kernel_adapter PRIVATE .)
target_link_libraries(video_kernel_adapter quad_kernel_bridge)
