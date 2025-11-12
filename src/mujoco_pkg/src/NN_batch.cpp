//範例
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>

int SAMPLE_DIM = 4;
int TARGET_DIM = 2;
int MEMORY_SIZE = 30;
int BATCH_SIZE = 16;
float LR = 1e-3;

// --- 1. Model 定義（簡單 MLP） ---
struct NetImpl : torch::nn::Module {
	torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr};
	NetImpl(int in_dim, int hidden, int out_dim) {
		l1 = register_module("l1", torch::nn::Linear(in_dim, hidden));
		l2 = register_module("l2", torch::nn::Linear(hidden, hidden));
		l3 = register_module("l3", torch::nn::Linear(hidden, out_dim));
	}
	torch::Tensor forward(torch::Tensor x) {
		x = torch::leaky_relu(l1->forward(x));
		x = torch::leaky_relu(l2->forward(x));
		return l3->forward(x);
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
		if (buffer_.size() >= MEMORY_SIZE) {
			buffer_.pop_front();
		}
		buffer_.push_back({sample, target});
	}
	std::vector<Experience> sample() {
		std::vector<Experience> batch; //宣告一個名稱叫做 batch 的變數，型態為 std::vector<Experience>
		std::vector<int> indices(buffer_.size());
		std::iota(indices.begin(), indices.end(), 0);
		std::shuffle(indices.begin(), indices.end(), rng);
		for (int i = 0; i < BATCH_SIZE; ++i) {
			batch.push_back(buffer_[indices[i]]);
		}
		return batch;
	}
	int size() {
		return buffer_.size();
	}
	
private:
	std::deque<Experience> buffer_; //宣告一個名稱叫做 buffer_ 的變數，型態為 std::deque<Experience>，頭尾兩端插入及刪除十分快速的陣列，元素型態為 Experience
	std::mt19937 rng{std::random_device{}()}; //建立一個亂數引擎
};

class deep_learning {
public:
	deep_learning() :
		device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
		net_1(SAMPLE_DIM, 128, TARGET_DIM),
		optimizer_net_1(net_1->parameters(), torch::optim::AdamOptions(LR))
	{
		net_1->to(device);
	}
	
	std::vector<float> net_1_forward(const std::vector<float>& sample) {
		torch::NoGradGuard no_grad;
		auto sample_tensor = torch::tensor(sample).reshape({1, SAMPLE_DIM}).to(device); //向量資料轉成 Tensor，並放在device上
		auto output_tensor = net_1->forward(sample_tensor); //net_1和sample_tensor都在device上，forward運算就會在device上運算
		auto output_cpu = output_tensor.to(torch::kCPU).squeeze();
		std::vector<float> output(output_cpu.data_ptr<float>(),
                                  output_cpu.data_ptr<float>() + output_cpu.numel());
		
		return output;
	}
	
	void update(ReplayBuffer &experience_repository) {
		if (experience_repository.size() < BATCH_SIZE) return;
		
		auto batch = experience_repository.sample();
		std::vector<float> all_sample, all_target;
		all_sample.reserve(BATCH_SIZE * SAMPLE_DIM);
		all_target.reserve(BATCH_SIZE * TARGET_DIM);
		
		for (const auto& e : batch) {
			for (int i = 0; i < SAMPLE_DIM; ++i) {
				all_sample.push_back(static_cast<float>(e.sample[i]));
			}
			for (int i = 0; i < TARGET_DIM; ++i) {
				all_target.push_back(static_cast<float>(e.target[i]));
			}
		}
//		std::cout << "all_sample: " << all_sample << std::endl;
		
		auto sample_tensor = torch::tensor(all_sample).reshape({BATCH_SIZE, SAMPLE_DIM}).to(device);
		auto target_tensor = torch::tensor(all_target).reshape({BATCH_SIZE, TARGET_DIM}).to(device);
//		std::cout << "sample_tensor: " << sample_tensor << std::endl;
		
		// ------ Backpropagate ------
		auto output = net_1->forward(sample_tensor); // Forward
		auto loss = torch::mse_loss(output, target_tensor); // 計算 Loss
		optimizer_net_1.zero_grad(); // 清空上一輪梯度
		loss.backward();       // 反向傳播
		torch::nn::utils::clip_grad_norm_(net_1->parameters(), 1.0); // 防止梯度爆炸
		optimizer_net_1.step();      // 更新權重
	}
	
private:
	torch::Device device;
	Net net_1{nullptr};
	torch::optim::Adam optimizer_net_1;
};

int main() {
	deep_learning deep_l;
	ReplayBuffer experience_repository;
	
	std::vector<float> sample = {10, 12, -30, 22};
	std::vector<float> target = {14, -54};
	
	std::vector<float> output = deep_l.net_1_forward(sample);
	std::cout << "Untrained output: " << output << std::endl;
	
	for (int i = 0; i < 50; i++) {
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
	
	for (int i = 0; i < 50; i++) {
		deep_l.update(experience_repository);
	}
	
	sample = {10, 12, -30, 22};
	target = {14, -54};
	output = deep_l.net_1_forward(sample);
	std::cout << "Trained output: " << output << std::endl;
	std::cout << "Target: " << target << std::endl;
	
	return 0;
}


