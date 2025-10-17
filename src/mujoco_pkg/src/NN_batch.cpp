//範例
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>

int input_dim = 4;
int hidden1 = 8;
int hidden2 = 4;
int output_dim = 2;

int MEMORY_SIZE = 10;
int BATCH_SIZE = 5;
float LR = 5e-2;

// --- 1. Model 定義（簡單 MLP） ---
struct Net : torch::nn::Module {
	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
	
	Net(int64_t input_dim, int64_t hidden1, int64_t hidden2, int64_t output_dim) {
		fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden1));
		fc2 = register_module("fc2", torch::nn::Linear(hidden1, hidden2));
		fc3 = register_module("fc3", torch::nn::Linear(hidden2, output_dim));
	}
	
	torch::Tensor forward(torch::Tensor x) {
		x = torch::leaky_relu(fc1->forward(x));
		x = torch::leaky_relu(fc2->forward(x));
		x = fc3->forward(x);
		return x;
	}
};

// ==================== Replay Buffer ========================
class ReplayBuffer {
public:
	//自訂資料型態
	struct Experience {
		std::vector<double> sample;
		std::vector<double> target;
	};
	
	void push(const std::vector<double>& sample, const std::vector<double>& target) { // const std::vector<double>& s 唯讀不複製 速度快
		if (buffer_.size() >= MEMORY_SIZE) buffer_.pop_front();
		buffer_.push_back({sample, target});
	}
	
	std::vector<Experience> sample() {
		std::vector<Experience> batch; //宣告一個名稱叫做 batch 的變數，型態為 std::vector<Experience>
		std::uniform_int_distribution<int> dist(0, buffer_.size() - 1); //dist(rng) 為0到buffer_.size() - 1隨機整數
		for (int i=0; i<BATCH_SIZE; i++) {
			batch.push_back(buffer_[dist(rng)]); // 有可能重複
		}
		return batch;
	}
	
	int size() const { return buffer_.size(); }
	
private:
	std::deque<Experience> buffer_; //宣告一個名稱叫做 buffer_ 的變數，型態為 std::deque<Experience>，頭尾兩端插入及刪除十分快速的陣列，元素型態為 Experience
	std::mt19937 rng{std::random_device{}()}; //建立一個亂數引擎
};

void train(Net& neural_network, 
				torch::optim::Optimizer& optimizer, 
				const std::vector<ReplayBuffer::Experience>& batch, 
				const torch::Device& device)
{
	int batch_size = batch.size();
	torch::nn::MSELoss loss_;
	
	std::vector<double> all_samples;
	std::vector<double> all_targets;
	
	for (const auto& exp : batch) {
		all_samples.insert(all_samples.end(), exp.sample.begin(), exp.sample.end());
		all_targets.insert(all_targets.end(), exp.target.begin(), exp.target.end());
	}
	
	torch::Tensor x = torch::tensor(all_samples).reshape({batch_size, input_dim}).to(device);; //向量資料轉成 PyTorch Tensor
	torch::Tensor t = torch::tensor(all_targets).reshape({batch_size, output_dim}).to(device);
	
	// ------ Forward ------
	torch::Tensor output = neural_network.forward(x);
	
	// ------ 計算 Loss ------
	torch::Tensor loss = loss_(output, t);
	
	// ------ Backpropagate ------
	optimizer.zero_grad(); // 清空上一輪梯度
	loss.backward();       // 反向傳播
	optimizer.step();      // 更新權重
}

int main() {
	torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	std::cout << "Using device: " << (device.is_cuda() ? "CUDA (GPU)" : "CPU") << std::endl;
	
	Net test_net{input_dim, hidden1, hidden2, output_dim};
	test_net.to(device);
	torch::optim::Adam optimizer(test_net.parameters(), torch::optim::AdamOptions(LR));
	
	ReplayBuffer experience_repository;
	
	std::vector<double> sample = {0.5, -1.2, 0.3, 0.8};
	std::vector<double> target = {8, 4};
	experience_repository.push(sample, target);
	
	torch::Tensor x = torch::tensor(sample).reshape({1, 4}); //向量資料轉成 PyTorch Tensor
	torch::Tensor output = test_net.forward(x);
	std::cout << "Forward output:\n" << output << std::endl;
	
	auto batch = experience_repository.sample();
	train(test_net, optimizer, batch, device);
	
	output = test_net.forward(x);
	std::cout << "Forward output:\n" << output << std::endl;
	
	return 0;
}









