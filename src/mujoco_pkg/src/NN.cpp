//範例
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>

// --- 1. Model 定義（簡單 MLP） ---
struct Net : torch::nn::Module {
	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
	
	Net(int64_t input_dim, int64_t hidden1, int64_t hidden2, int64_t output_dim) {
		fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden1));
		fc2 = register_module("fc2", torch::nn::Linear(hidden1, hidden2));
		fc3 = register_module("fc3", torch::nn::Linear(hidden2, output_dim));
	}
	
	torch::Tensor forward(std::vector<float> sample) {
		torch::Tensor x = torch::tensor(sample).reshape({1, 4}); //向量資料轉成 PyTorch Tensor
		x = torch::leaky_relu(fc1->forward(x));
		x = torch::leaky_relu(fc2->forward(x));
		x = fc3->forward(x);
		return x;
	}
};

void train(Net& neural_network, 
				torch::optim::Optimizer& optimizer, 
				const std::vector<float> sample, 
				const std::vector<float> target, 
				const torch::Device& device)
{
	torch::nn::MSELoss loss_;
	
	torch::Tensor t = torch::tensor(target).reshape({1, 2}); //向量資料轉成 PyTorch Tensor
	
	// ------ Forward ------
	torch::Tensor output = neural_network.forward(sample);
	
	// ------ 計算 Loss ------
	torch::Tensor loss = loss_(output, t);
	
	// ------ Backpropagate ------
	optimizer.zero_grad(); // 清空上一輪梯度
	loss.backward();       // 反向傳播
	optimizer.step();      // 更新權重
}

int main() {
	int input_dim = 4;
	int hidden1 = 8;
	int hidden2 = 4;
	int output_dim = 2;
	float LR = 5e-2;
	
	torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	std::cout << "Using device: " << (device.is_cuda() ? "CUDA (GPU)" : "CPU") << std::endl;
	
	Net test_net{input_dim, hidden1, hidden2, output_dim};
	test_net.to(device);
	torch::optim::Adam optimizer(test_net.parameters(), torch::optim::AdamOptions(LR));
	
//	std::cout << policy_net.fc1->weight << std::endl;
//	std::cout << policy_net.fc1->bias << std::endl;
	
	std::vector<float> sample = {0.5, -1.2, 0.3, 0.8};
	torch::Tensor output = test_net.forward(sample);
	std::cout << "Forward output:\n" << output << std::endl;
	
	std::vector<float> target = {8, 4};
	
	for (int i = 1; i < 100; i++) {
		train(test_net, optimizer, sample, target, device);
	}
	
	output = test_net.forward(sample);
	std::cout << "Forward output:\n" << output << std::endl;
	
	return 0;
}









