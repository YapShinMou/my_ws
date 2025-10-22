//範例
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>

int NN_input_dim = 4;
int NN_output_dim = 2;

int MEMORY_SIZE = 100;
int BATCH_SIZE = 32;
float LR = 1e-3;

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

// ==================== Replay Buffer ========================
class ReplayBuffer {
public:
	//自訂資料型態
	struct Experience {
		std::vector<float> sample;
		std::vector<float> target;
	};
	
	void push(const std::vector<float>& sample, const std::vector<float>& target) { // const std::vector<double>& s 唯讀不複製 速度快
		if (buffer_.size() >= MEMORY_SIZE) buffer_.pop_front();
		buffer_.push_back({sample, target});
	}
	
	std::vector<Experience> sample() {
		std::vector<Experience> batch; //宣告一個名稱叫做 batch 的變數，型態為 std::vector<Experience>
		std::vector<int> indices(buffer_.size());
		std::iota(indices.begin(), indices.end(), 0);
		std::shuffle(indices.begin(), indices.end(), rng);
		for (int i = 0; i < BATCH_SIZE && i < (int)indices.size(); ++i) {
			batch.push_back(buffer_[indices[i]]);
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
	
	std::vector<float> all_samples;
	std::vector<float> all_targets;
	
	all_samples.reserve(batch_size * NN_input_dim);
	all_targets.reserve(batch_size * NN_output_dim);
	
	for (const auto& e : batch) {
		for (int i = 0; i < NN_input_dim; ++i) {
			all_samples.push_back(static_cast<float>(e.sample[i]));
		}
		for (int i = 0; i < NN_output_dim; ++i) {
			all_targets.push_back(static_cast<float>(e.target[i]));
		}
	}
	
	auto sample_tensor = torch::tensor(all_samples).reshape({batch_size, NN_input_dim}).to(device); //向量資料轉成 PyTorch Tensor
	auto target_tensor = torch::tensor(all_targets).reshape({batch_size, NN_output_dim}).to(device);
	
	// ------ Backpropagate ------
	auto output = neural_network->forward(sample_tensor); // Forward
	auto loss = loss_(output, target_tensor); // 計算 Loss
	optimizer.zero_grad(); // 清空上一輪梯度
	loss.backward();       // 反向傳播
	torch::nn::utils::clip_grad_norm_(neural_network->parameters(), 1.0); // 防止梯度爆炸
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
	// optimizer 要追蹤梯度，optimizer 放在train()裡面會把訓練重置掉
	torch::optim::Adam optimizer(test_net->parameters(), torch::optim::AdamOptions(LR));
	
	ReplayBuffer experience_repository;
	
	std::vector<float> sample = {10, 12, -30, 22};
	std::vector<float> target = {14, -54};
	torch::Tensor x = torch::tensor(sample).reshape({1, 4}); //向量資料轉成 PyTorch Tensor
	torch::Tensor output = test_net->forward(x);
	std::cout << "Untrained output:\n" << output << std::endl;
	
	for (int i = 0; i < 500; i++) {
		std::mt19937 rng{std::random_device{}()};
		std::uniform_int_distribution<int> random_1(-50, 50);
		std::uniform_int_distribution<int> random_2(-50, 50);
		std::uniform_int_distribution<int> random_3(-50, 50);
		std::uniform_int_distribution<int> random_4(-50, 50);
		
		float sample_1 = random_1(rng);
		float sample_2 = random_2(rng);
		float sample_3 = random_3(rng);
		float sample_4 = random_4(rng);
		
		sample = {sample_1, sample_2, sample_3, sample_4};
		target = {sample_1+sample_2+sample_3+sample_4, sample_1-sample_2+sample_3-sample_4};
//		std::cout << sample << " " << target << std::endl;
		experience_repository.push(sample, target);
	}
	
	for (int i = 0; i < 100; i++) {
		std::vector<ReplayBuffer::Experience> batch = experience_repository.sample();
		train(test_net, optimizer, batch, device);
	}
	
	sample = {10, 12, -30, 22};
	target = {14, -54};
	x = torch::tensor(sample).reshape({1, 4});
	auto t = torch::tensor(target).reshape({1, 2});
	output = test_net->forward(x);
	std::cout << "Trained output:\n" << output << std::endl;
	std::cout << "Target:\n" << t << std::endl;
	
	return 0;
}









