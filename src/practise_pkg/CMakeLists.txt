cmake_minimum_required(VERSION 3.10)
project(practise_pkg)

include_directories(include)

add_executable(constructor_lib_example src/constructor_lib_example.cpp src/user_def_lib.cpp)

