cmake_minimum_required(VERSION 3.23)
project(entregable_1 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(entregable_1 main.cu)

set_target_properties(entregable_1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
