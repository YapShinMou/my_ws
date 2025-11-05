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
	std::vector<float> sample = {1, -0.5, 0, 0.7};
	auto policy_output = torch::tensor(sample).reshape({1, ACTION_DIM * 2});
	std::cout << policy_output << std::endl;
	auto mean = policy_output.slice(1, 0, ACTION_DIM);
	auto log_sigma = policy_output.slice(1, ACTION_DIM, 2 * ACTION_DIM);
	auto sigma_ = log_sigma.exp();
	auto action_normal = mean + sigma_ * torch::randn_like(mean);
	auto action_cpu_tensor = torch::tanh(action_normal).to(torch::kCPU);
	std::vector<float> action(action_cpu_tensor.data_ptr<float>(),
                                  action_cpu_tensor.data_ptr<float>() + action_cpu_tensor.numel());
	
	std::cout << mean << std::endl;
	std::cout << log_sigma << std::endl;
	std::cout << sigma_ << std::endl;
	std::cout << action_normal << std::endl;
	std::cout << action_cpu_tensor << std::endl;
	std::cout << action << std::endl;
}
