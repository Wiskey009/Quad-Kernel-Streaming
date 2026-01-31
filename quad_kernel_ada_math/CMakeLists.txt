# CMake wrapper for Ada Math Kernel
# We treat the Alire-built library as an IMPORTED static library

# Path to the library built by Alire
set(ADA_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib/libquad_math.a")

add_library(math_kernel_adapter STATIC IMPORTED)

# Set the location of the imported library
set_target_properties(math_kernel_adapter PROPERTIES
    IMPORTED_LOCATION "${ADA_LIB_PATH}"
)

# We might need to link against the GNAT runtime libraries if linking a C main,
# generally passed via linker flags in the main project or handled by the compiler.
# For MinGW/GNAT, usually -lgnat -lgnarl are needed.

target_include_directories(math_kernel_adapter INTERFACE .)
