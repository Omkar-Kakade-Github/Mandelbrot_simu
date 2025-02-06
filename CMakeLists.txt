cmake_minimum_required(VERSION 3.10)
project(CUDA_Mandelbrot LANGUAGES CXX CUDA)

# Find OpenGL, GLEW, and GLFW
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(CUDA REQUIRED)

# Include necessary directories
include_directories(${GLEW_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS} /usr/local/cuda/include)

# Specify CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75) 

# Add the executable
add_executable(CUDA_Mandelbrot
    src/main.cpp 
    src/shader_utils.cpp  
    src/mandelbrot_kernel.cu
)

# Link libraries
target_link_libraries(CUDA_Mandelbrot ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} glfw ${CUDA_LIBRARIES})
