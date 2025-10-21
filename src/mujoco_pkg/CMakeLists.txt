cmake_minimum_required(VERSION 3.10)
project(mujoco_pkg)

set(Torch_DIR /home/yap/libtorch/share/cmake/Torch) #自己新增###############################
find_package(Torch REQUIRED) #自己新增###############################

include_directories(/usr/local/include/mujoco)
link_directories(/usr/local/lib/mujoco)

add_definitions(-DMUJOCO_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models")

add_executable(CartPole src/CartPole.cpp)
add_executable(CartPole_without_GUI src/CartPole_without_GUI.cpp)
add_executable(Dog src/Dog.cpp)
add_executable(CartPole_DQN src/CartPole_DQN.cpp)
add_executable(CartPole_DQN_without_GUI src/CartPole_DQN_without_GUI.cpp)
add_executable(CartPole_DQN_test src/CartPole_DQN_test.cpp)
add_executable(test src/test.cpp)
add_executable(NN src/NN.cpp)
add_executable(NN_batch src/NN_batch.cpp)
add_executable(DQN_test src/DQN_test.cpp)

target_link_libraries(CartPole mujoco glfw pthread)
target_link_libraries(CartPole_without_GUI mujoco glfw GL m dl pthread)
target_link_libraries(Dog mujoco glfw GL m dl pthread)
target_link_libraries(CartPole_DQN mujoco glfw GL m dl pthread ${TORCH_LIBRARIES})
target_link_libraries(CartPole_DQN_without_GUI mujoco glfw GL m dl pthread ${TORCH_LIBRARIES})
target_link_libraries(CartPole_DQN_test mujoco glfw GL m dl pthread ${TORCH_LIBRARIES})
target_link_libraries(test mujoco glfw GL m dl pthread ${TORCH_LIBRARIES})
target_link_libraries(NN mujoco glfw GL m dl pthread ${TORCH_LIBRARIES})
target_link_libraries(NN_batch mujoco glfw GL m dl pthread ${TORCH_LIBRARIES})
target_link_libraries(DQN_test mujoco glfw GL m dl pthread ${TORCH_LIBRARIES})

