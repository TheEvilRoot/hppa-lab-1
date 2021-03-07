cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)
set(SOURCE_FILES main.cc)

project(vectorization CXX)

add_executable(no_vectorize ${SOURCE_FILES})
target_compile_options(no_vectorize PRIVATE -O3 -fno-vectorize -fno-slp-vectorize)

add_executable(vectorize ${SOURCE_FILES})
target_compile_options(vectorize PRIVATE -O3 -fvectorize -ftree-vectorize -fslp-vectorize -ftree-slp-vectorize)

