cmake_minimum_required(VERSION 3.10)
project(mujoco_pkg)

include_directories(/usr/local/include/mujoco)
link_directories(/usr/local/lib/mujoco)

add_executable(CartPole src/CartPole.cpp)
add_executable(CartPole_without_GUI src/CartPole_without_GUI.cpp)

target_link_libraries(CartPole mujoco glfw GL m dl pthread)
target_link_libraries(CartPole_without_GUI mujoco glfw GL m dl pthread)
