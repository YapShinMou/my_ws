#include <string>
#include <thread>
#include <mutex>
#include <vector>
#include <deque>
#include <random>
#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

int STATE_DIM = 4;
int ACTION_DIM = 2;

int main() {
	torch::Tensor tensor = torch::eye(3);
	std::cout << tensor << std::endl;
}
