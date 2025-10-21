//範例
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>

int state_dim = 4;
int action_dim = 2;
int MEMORY_SIZE = 10;
int BATCH_SIZE = 4;
float LR = 1e-3;

int EPISODES = 5;
float GAMMA = 0.9;
int TARGET_UPDATE = 2;

// --- 1. Model 定義（簡單 MLP） ---
struct NetImpl : torch::nn::Module {
	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
	
	NetImpl() {
		fc1 = register_module("fc1", torch::nn::Linear(state_dim, 128));
		fc2 = register_module("fc2", torch::nn::Linear(128, 64));
		fc3 = register_module("fc3", torch::nn::Linear(64, action_dim));
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
		std::vector<float> state;
		int action;
		float reward;
		std::vector<float> next_state;
		bool done;
	};
	
	// const std::vector<float>& s 唯讀不複製 速度快
	void push(const std::vector<float>& state, int action, float reward, const std::vector<float>& next_state, bool done) { 
		if (buffer_.size() >= MEMORY_SIZE) buffer_.pop_front();
		buffer_.push_back({state, action, reward, next_state, done});
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

void train(Net& policy_net, 
				Net& target_net, 
				torch::optim::Optimizer& optimizer, 
				const std::vector<ReplayBuffer::Experience>& batch, 
				const torch::Device& device)
{
	int batch_size = batch.size();
	torch::nn::MSELoss loss_;
	
	std::vector<double> all_state, all_next_state;
	std::vector<int> all_action;
	std::vector<float> all_reward;
	std::vector<float> all_done;
	
	all_state.reserve(batch_size * state_dim);
	all_action.reserve(batch_size);
	all_reward.reserve(batch_size);
	all_next_state.reserve(batch_size * state_dim);
	all_done.reserve(batch_size);
	
	for (const auto& e : batch) {
		for (int i = 0; i < state_dim; ++i) {
			all_state.push_back(static_cast<float>(e.state[i]));
			all_next_state.push_back(static_cast<float>(e.next_state[i]));
		}
		all_action.push_back(e.action);
		all_reward.push_back(e.reward);
		all_done.push_back(e.done ? 0.0f : 1.0f);  // done=true -> 0
	}
	
	auto state_tensor = torch::tensor(all_state).reshape({batch_size, state_dim}).to(device); //向量資料轉成 PyTorch Tensor
	auto action_tensor = torch::tensor(all_action, torch::kInt64).to(device);
	auto reward_tensor = torch::tensor(all_reward).to(device);
	auto next_state_tensor = torch::tensor(all_next_state).reshape({batch_size, state_dim}).to(device);
	auto done_tensor = torch::tensor(all_done).to(device);
	
	auto q_values_all = policy_net->forward(state_tensor); // 取出各個狀態下的所有Q值
	auto q_values = q_values_all.gather(1, action_tensor.unsqueeze(1)).squeeze(1); // 只留下action相對應的Q(s, a)
	
	auto next_q_all = target_net->forward(next_state_tensor);
	auto next_q_values = std::get<0>(next_q_all.max(1)); // max(Q)
	
	auto expected_q = reward_tensor + GAMMA * next_q_values * done_tensor;
	std::cout << next_q_all << std::endl;
	std::cout << expected_q << std::endl;
	
	auto loss = loss_(q_values, expected_q.detach()); // .detach() 避免更新 target_net
	
	// ------ Backpropagate ------
	optimizer.zero_grad(); // 清空上一輪梯度
	loss.backward();       // 反向傳播
	torch::nn::utils::clip_grad_norm_(policy_net->parameters(), 1.0); // 防止梯度爆炸
	optimizer.step();      // 更新權重
}

int main() {
	// -----------------------------------
	// 初始神經網路
	// -----------------------------------
	torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	std::cout << "Using device: " << (device.is_cuda() ? "CUDA (GPU)" : "CPU") << std::endl;
	
	auto policy_net = Net();
	auto target_net = Net();
	policy_net->to(device);
	target_net->to(device);
	// optimizer 要追蹤梯度，optimizer 放在train()裡面會把訓練重置掉
	torch::optim::Adam optimizer(policy_net->parameters(), torch::optim::AdamOptions(LR));
	
	torch::save(policy_net, "tmp.pt");
	torch::load(target_net, "tmp.pt");
	target_net->eval(); //設定成推論模式
	
	ReplayBuffer memory;
	
	// -----------------------------------
	// 主迴圈
	// -----------------------------------
	for (int episode = 0; episode < EPISODES ; ++episode)
	{
		bool done = false;
		int step_count = 0;
		std::vector<float> state = {5, 0, 10, 0};
		
		while (!done) {
			int action = 1;
			std::vector<float> next_state = {10, 5, 10, 5};
			
			float reward = 1;
			
			step_count = step_count + 1;
			if (step_count>10) done = true;
			
			// store to memory
			memory.push(state, action, reward, next_state, done);
			
			// update state
//			state = next_state;
			
			// train if enough samples
			if (memory.size() >= BATCH_SIZE) {
				std::vector<ReplayBuffer::Experience> batch = memory.sample();
				train(policy_net, target_net, optimizer, batch, device);
			}
		} // end episode loop
		
		if (episode % TARGET_UPDATE == 0) {
			torch::save(policy_net, "tmp.pt");
			torch::load(target_net, "tmp.pt");
			std::cout << "Episode " << episode << " update target net" << std::endl;
		}
	} // end training loop
	return 0;
}









