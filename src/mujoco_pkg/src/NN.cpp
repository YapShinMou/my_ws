//範例
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>

int NN_input_dim = 4;
int NN_output_dim = 2;
float LR = 5e-2; // 神經網路學習率

// --- 1. Model 定義（簡單 MLP） ---
struct NetImpl : torch::nn::Module {
	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
	NetImpl() {
		fc1 = register_module("fc1", torch::nn::Linear(NN_input_dim, 128));
		fc2 = register_module("fc2", torch::nn::Linear(128, 64));
		fc3 = register_module("fc3", torch::nn::Linear(64, NN_output_dim));
	}
	torch::Tensor forward(torch::Tensor x) {
		x = torch::leaky_relu(fc1->forward(x));
		x = torch::leaky_relu(fc2->forward(x));
		x = fc3->forward(x);
		return x;
	}
};
TORCH_MODULE(Net);

void train(Net& neural_network, 
				torch::optim::Optimizer& optimizer, 
				const std::vector<float>& sample, 
				const std::vector<float>& target, 
				const torch::Device& device)
{
	torch::nn::MSELoss loss_;
	
	auto sample_tensor = torch::tensor(sample).reshape({1, 4}); //向量資料轉成 Tensor
	auto target_tensor = torch::tensor(target).reshape({1, 2});
	
	// ------ Backpropagate ------
	auto output = neural_network->forward(sample_tensor); // Forward
	auto loss = loss_(output, target_tensor); // 計算 Loss
	optimizer.zero_grad(); // 清空上一輪梯度
	loss.backward();       // 反向傳播
	optimizer.step();      // 更新權重
}

int main() {
	// -----------------------------------
	// 初始神經網路
	// -----------------------------------
	torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	std::cout << "Using device: " << (device.is_cuda() ? "CUDA (GPU)" : "CPU") << std::endl;
	
	auto test_net = Net();
	test_net->to(device);
//	optimizer 要追蹤梯度，optimizer 放在train()裡面會把訓練重置掉
	torch::optim::Adam optimizer(test_net->parameters(), torch::optim::AdamOptions(LR));
	
//	std::cout << test_net->fc1->weight << std::endl;
//	std::cout << test_net->fc1->bias << std::endl;
	
	std::vector<float> sample = {10, 12, -30, 22};
	auto x = torch::tensor(sample).reshape({1, 4});
	auto output = test_net->forward(x);
	std::cout << "Untrained output:\n" << output << std::endl;
	
	std::vector<float> target = {14, -54};
	auto t = torch::tensor(target).reshape({1, 2});
	
	for (int i = 1; i < 100; i++) {
		train(test_net, optimizer, sample, target, device);
	}
	
	output = test_net->forward(x);
	std::cout << "Trained output:\n" << output << std::endl;
	std::cout << "Target:\n" << t << std::endl;
	return 0;
}
