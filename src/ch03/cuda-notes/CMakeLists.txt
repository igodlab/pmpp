cmake_minimum_required(VERSION 3.28)
project(Ch03 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})  # point to pixi-managed packages

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)  

# -----------------------------------------------------------------------
# utils as a static library — compiled ONCE, linked into every binary
# -----------------------------------------------------------------------
add_library(utils STATIC utils.cpp)

target_include_directories(utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}   # utils.hpp is visible to anything linking utils
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(utils PUBLIC
    ${OpenCV_LIBS}
)

# -----------------------------------------------------------------------
# Helper function — avoids repeating boilerplate for every new script
# -----------------------------------------------------------------------
function(add_cuda_exercise target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE utils CUDA::cudart)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endfunction()

# -----------------------------------------------------------------------
# Register each script — adding a new one is just one line
# -----------------------------------------------------------------------
add_cuda_exercise(vector_addition  vector_addition.cu)
add_cuda_exercise(color_to_grey    color_to_grey.cu)
add_cuda_exercise(image_blur    image_blur.cu)
add_cuda_exercise(matrix_multiplication matrix_multiplication.cu)
