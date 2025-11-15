//範例
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>

// ====== 超參數 ======
const int STATE_DIM = 4;
const int ACTION_DIM = 2;
const int MEMORY_SIZE = 10;
const int BATCH_SIZE = 4;
const int EPISODES = 10;
const int TIME_STEPS = 5;

const float GAMMA = 0.99;
const float TAU = 0.005;
const float LR_Q = 3e-4;
const float LR_PI = 3e-4;
const float LR_ALPHA = 3e-4;

// ========= MLP ===========
struct MLPImpl : torch::nn::Module {
	torch::nn::Linear layer1{nullptr}, layer2{nullptr}, layer3{nullptr};
	
	MLPImpl(int in_dim, int hidden, int out_dim) {
		layer1 = register_module("layer1", torch::nn::Linear(in_dim, hidden));
		layer2 = register_module("layer2", torch::nn::Linear(hidden, hidden));
		layer3 = register_module("layer3", torch::nn::Linear(hidden, out_dim));
	}
	
	torch::Tensor forward(torch::Tensor x) {
		x = torch::relu(layer1->forward(x));
		x = torch::relu(layer2->forward(x));
		return layer3->forward(x);
	}
};
TORCH_MODULE(MLP);

// ==================== Replay Buffer ========================
class ReplayBuffer {
public:
	//自訂資料型態
	struct Experience {
		std::vector<float> state, action;
		float reward;
		std::vector<float> next_state;
		float not_terminal;
	};
	
	void push(const std::vector<float>& s, 
					const std::vector<float>& a, 
					const float& r, 
					const std::vector<float>& s2, 
					const float& not_t) // const 唯讀，& 不複製，速度快
	{
		if (buffer_.size() >= MEMORY_SIZE) {
			buffer_.pop_front();
		}
		buffer_.push_back({s, a, r, s2, not_t});
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

// ============= SAC Agent =============
class SACAgent {
public:
	SACAgent() :
		device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
		
		q1_net(STATE_DIM + ACTION_DIM, 4, 1),
		q2_net(STATE_DIM + ACTION_DIM, 4, 1),
		q1_target(STATE_DIM + ACTION_DIM, 4, 1),
		q2_target(STATE_DIM + ACTION_DIM, 4, 1),
		policy_net(STATE_DIM, 4, 2 * ACTION_DIM),
		log_alpha(torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(device))),
		
		opt_q1(q1_net->parameters(), torch::optim::AdamOptions(LR_Q)),
		opt_q2(q2_net->parameters(), torch::optim::AdamOptions(LR_Q)),
		opt_pi(policy_net->parameters(), torch::optim::AdamOptions(LR_PI)),
		opt_log_alpha(std::vector<torch::Tensor>{log_alpha}, torch::optim::AdamOptions(LR_ALPHA))
	{
		q1_net->to(device);
		q2_net->to(device);
		q1_target->to(device);
		q2_target->to(device);
		policy_net->to(device);
		
		torch::load(policy_net, "policy_net.pt");
	}
	
	std::vector<float> select_action(const std::vector<float>& state) {
		torch::NoGradGuard no_grad;
		auto state_tensor = torch::tensor(state).reshape({1, STATE_DIM}).to(device); //向量資料轉成 Tensor，並放在device上
		auto [action_tensor, next_log_pi] = sample_action(state_tensor);
		auto action_cpu = action_tensor.to(torch::kCPU).squeeze();
		std::vector<float> action(action_cpu.data_ptr<float>(),
                                  action_cpu.data_ptr<float>() + action_cpu.numel());
		std::cout << "select_action: " << action << std::endl;
		std::cout << " " << std::endl;
		return action;
	}
	
	void update(ReplayBuffer &memory) {
		if (memory.size() < BATCH_SIZE) return;
		
		std::cout << "--- update() ---" << std::endl;
		auto batch = memory.sample();
		std::vector<float> all_state, all_action, all_reward, all_next_state, all_not_t;
		all_state.reserve(BATCH_SIZE * STATE_DIM);
		all_action.reserve(BATCH_SIZE * ACTION_DIM);
		all_reward.reserve(BATCH_SIZE);
		all_next_state.reserve(BATCH_SIZE * STATE_DIM);
		all_not_t.reserve(BATCH_SIZE);
		
		for (const auto& e : batch) {
			for (int i = 0; i < STATE_DIM; ++i) {
				all_state.push_back(static_cast<float>(e.state[i]));
				all_next_state.push_back(static_cast<float>(e.next_state[i]));
			}
			for (int i = 0; i < ACTION_DIM; ++i) {
				all_action.push_back(static_cast<float>(e.action[i]));
			}
			all_reward.push_back(e.reward);
			all_not_t.push_back(e.not_terminal);
		}
		
		auto state_tensor = torch::tensor(all_state).reshape({BATCH_SIZE, STATE_DIM}).to(device);
		auto action_tensor = torch::tensor(all_action).reshape({BATCH_SIZE, ACTION_DIM}).to(device);
		auto reward_tensor = torch::tensor(all_reward).reshape({BATCH_SIZE, 1}).to(device);
		auto next_state_tensor = torch::tensor(all_next_state).reshape({BATCH_SIZE, STATE_DIM}).to(device);
		auto not_t_tensor = torch::tensor(all_not_t).reshape({BATCH_SIZE, 1}).to(device);
//		std::cout << "state_tensor: " << state_tensor << std::endl;
//		std::cout << "action_tensor: " << action_tensor << std::endl;
//		std::cout << "reward_tensor: " << reward_tensor << std::endl;
//		std::cout << "next_state_tensor: " << next_state_tensor << std::endl;
//		std::cout << "not_t_tensor: " << not_t_tensor << std::endl;
		
		auto alpha = log_alpha.exp();
//		std::cout << "alpha :" << alpha << std::endl;
		
		// --- Q target ---
		std::cout << "--- Q target ---" << std::endl;
		auto [next_action, next_log_pi] = sample_action(next_state_tensor);
		auto next_q1 = q1_target->forward(torch::cat({next_state_tensor, next_action}, 1));
		auto next_q2 = q2_target->forward(torch::cat({next_state_tensor, next_action}, 1));
//		std::cout << "next_q1 :" << next_q1 << std::endl;
//		std::cout << "next_q2 :" << next_q2 << std::endl;
		auto q_target_min = torch::min(next_q1, next_q2);
//		std::cout << "q_target_min :" << q_target_min << std::endl;
		auto q_target = reward_tensor + GAMMA * not_t_tensor * (q_target_min - alpha * next_log_pi);
//		std::cout << "q_target :" << q_target << std::endl; //correct

//		std::vector<float> q_target_test = {15, 15, 15, 15}; //test
//		q_target = torch::tensor(q_target_test).reshape({BATCH_SIZE, 1}).to(device); //test
//		std::cout << "q_target :" << q_target << std::endl;
		
		// --- Q updates ---
		auto q_input = torch::cat({state_tensor, action_tensor}, 1);
		auto q1 = q1_net->forward(q_input);
		auto q2 = q2_net->forward(q_input);
//		std::cout << "q1 :" << q1 << std::endl;
//		std::cout << "q2 :" << q2 << std::endl;
		
		auto loss_q1 = torch::mse_loss(q1, q_target.detach());
		auto loss_q2 = torch::mse_loss(q2, q_target.detach());
		
		opt_q1.zero_grad(); loss_q1.backward(); opt_q1.step();
		opt_q2.zero_grad(); loss_q2.backward(); opt_q2.step(); //learnable
		
		// --- Policy update ---
		auto [a_pi, log_pi] = sample_action(state_tensor);
		auto q_input_pi = torch::cat({state_tensor, a_pi}, 1);
		auto q_min_pi = torch::min(q1_net->forward(q_input_pi), q2_net->forward(q_input_pi));
//		std::cout << "q_min_pi :" << q_min_pi << std::endl;
//		std::cout << "alpha :" << alpha << std::endl;
//		std::cout << "log_pi :" << log_pi << std::endl;
		auto loss_pi = (q_min_pi - alpha * log_pi).mean(); //currect
		
		opt_pi.zero_grad();
		loss_pi.backward();
		opt_pi.step();
		
		// --- alpha ---
		auto alpha_loss = -(log_alpha * (log_pi.detach() - ACTION_DIM)).mean();
		
		opt_log_alpha.zero_grad();
		alpha_loss.backward();
		opt_log_alpha.step();
		
		// --- Q-target update ---
		update_target_net();
	}
	
	void update_target_net() { //correct
		std::cout << "update_target_net()" << std::endl;
		torch::NoGradGuard no_grad;
		for (auto &pair : q1_target->named_parameters()) {
			
			auto name = pair.key();                            // 取參數名稱（例如 "layer1.weight"）
			std::cout << "name:" << name << std::endl;
			auto &tgt = pair.value();                          // 取得 target net 的參數 Tensor
			std::cout << "tgt:" << tgt << std::endl;
			auto &src = q1_net->named_parameters()[name];      // 對應到 q1_net 的同名參數 Tensor
			std::cout << "src:" << src << std::endl;
			tgt.copy_(src * (1.0 - TAU) + tgt * TAU);          // 按公式更新 target 參數
			std::cout << "tgt:" << tgt << std::endl;
		}
		
		for (auto &pair : q2_target->named_parameters()) {
			auto name = pair.key();
			auto &tgt = pair.value();
			auto &src = q2_net->named_parameters()[name];
			tgt.copy_(src * (1.0 - TAU) + tgt * TAU);
		}
	}
	
	void save_net() {
		torch::save(policy_net, "policy_net.pt");
	}
	
	void show_net() {
		std::cout << policy_net->layer1->weight << std::endl;
		std::cout << policy_net->layer1->bias << std::endl;
		std::cout << policy_net->layer2->weight << std::endl;
		std::cout << policy_net->layer2->bias << std::endl;
		std::cout << policy_net->layer3->weight << std::endl;
		std::cout << policy_net->layer3->bias << std::endl;
	}
	
private:
	torch::Device device;
	
	MLP q1_net{nullptr};
	MLP q2_net{nullptr};
	MLP q1_target{nullptr};
	MLP q2_target{nullptr};
	MLP policy_net{nullptr};
	torch::Tensor log_alpha;
	
	torch::optim::Adam opt_q1;
	torch::optim::Adam opt_q2;
	torch::optim::Adam opt_pi;
	torch::optim::Adam opt_log_alpha;
	
	const float ln_2pi = std::log(2 * M_PI);
	
	std::tuple<torch::Tensor, torch::Tensor> sample_action(const torch::Tensor& state) {
		std::cout << "sample_action()" << std::endl;
		auto mean_logstd = policy_net->forward(state);
//		std::vector<float> test_mean_lnsigma = {-10, 10, std::log(1), std::log(0.5)}; // test
//		mean_logstd = torch::tensor(test_mean_lnsigma).reshape({1, ACTION_DIM * 2}).to(device); // test
		
		auto mean = mean_logstd.slice(1, 0, ACTION_DIM);
		auto ln_sigma = torch::clamp(mean_logstd.slice(1, ACTION_DIM, 2 * ACTION_DIM), -20, 3);
		auto sigma_ = ln_sigma.exp();
//		std::cout << "mean: " << mean << std::endl;
//		std::cout << "sigma_: " << sigma_ << std::endl;
		
		auto eps = torch::randn_like(mean);
		auto z = mean + sigma_ * eps;
//		std::cout << "z: " << z << std::endl;
		auto action = torch::tanh(z);
		
		// log π(a|s)
		auto ln_pi_individual = -0.5 * (((z - mean) / sigma_).pow(2) + 2*ln_sigma + ln_2pi);
//		std::cout << "ln_pi_individual: " << ln_pi_individual << std::endl;
		
		auto ln_pi = ln_pi_individual.sum(1, true);
//		std::cout << "ln_pi: " << ln_pi << std::endl;
		
		auto ln_pi_tanh = ln_pi - (2 * (torch::log(torch::tensor(2.0, state.options())) - z - torch::nn::functional::softplus(-2 * z))).sum(1, true); //not sure
//		std::cout << "ln_pi_tanh_correction: " << ln_pi_tanh << std::endl;
		return std::make_tuple(action, ln_pi_tanh);
	}
};

float get_reward(const std::vector<float>& state, 
						const std::vector<float>& next_state, 
						const float& not_terminal)
{
	return 1;
}

int main() {
	SACAgent agent;
	ReplayBuffer memory;
	
	agent.show_net();
	
	for (int episode = 0; episode < EPISODES; ++episode) {
		float total_reward = 0;
		int step_count = 0;
		float not_terminal = 1;
		std::vector<float> state = {0, 0, 0, 0};
		
		while (not_terminal) {
			auto action = agent.select_action(state);
			
			// 模擬一步
			std::vector<float> next_state = {0, 0, 0, 0};
			
			step_count = step_count + 1;
			if (step_count > TIME_STEPS) not_terminal = 0;
			
			float reward = get_reward(state, next_state, not_terminal);
			total_reward = total_reward + reward;
			
			memory.push(state, action, reward, next_state, not_terminal);
			agent.update(memory);
			
			state = next_state;
		} // end episode loop
		std::cout << "Episode " << episode << ", total reward = " << total_reward << std::endl;
		std::cout << " " << std::endl;
	} // end training loop
	return 0;
}


