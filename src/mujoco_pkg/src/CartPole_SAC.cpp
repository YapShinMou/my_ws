// SAC_CartPole.cpp
#include <vector>
#include <deque>
#include <random>
#include <iostream>
#include <torch/torch.h>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

// MuJoCo data structures
mjModel* m = nullptr;                  // MuJoCo model
mjData* d = nullptr;                   // MuJoCo data
mjvCamera cam;                      // abstract camera +++
mjvOption opt;                      // visualization options +++
mjvScene scn;                       // abstract scene +++
mjrContext con;                     // custom GPU context +++
mjvPerturb pert; // +++

// ---------------- 超參數 ----------------
constexpr int STATE_DIM = 4;
constexpr int ACTION_DIM = 1;
constexpr int MEMORY_SIZE = 1000;
constexpr int BATCH_SIZE = 256;
constexpr int EPISODES = 1000;

float GAMMA = 0.99;
float TAU = 0.005;
float lr_q = 1e-4;
float lr_pi = 1e-6;
float lr_alpha = 3e-4;

float ALPHA = 0.001;

int ACTION_PARAM = 0;
int GLFW_SHOW = 0;
int SAVE_NET = 0;
int LOSS_POLICY = 0;
int LOSS_Q = 0;
int Q_MIN = 0;

// ---------------- 定義 MLP ----------------
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

// ---------------- Replay Buffer ----------------
class ReplayBuffer {
public:
	struct Experience {
		std::vector<float> state, action;
		float reward;
		std::vector<float> next_state;
		float not_terminal;
	};
	void push(std::vector<float> s, std::vector<float> a, float r, std::vector<float> s2, float not_t) {
		if (buffer_.size() >= MEMORY_SIZE) {
			buffer_.pop_front();
		}
		buffer_.push_back({s, a, r, s2, not_t});
	}
	std::vector<Experience> sample() {
		std::vector<Experience> batch;
		std::vector<int> indices(buffer_.size());
		std::iota(indices.begin(), indices.end(), 0);
		std::shuffle(indices.begin(), indices.end(), rng);
		for (int i = 0; i < BATCH_SIZE; i++) {
			batch.push_back(buffer_[indices[i]]);
		}
		return batch;
	}
	int size() {
		return buffer_.size();
	}
	
private:
	std::deque<Experience> buffer_;
	std::mt19937 rng{std::random_device{}()};
};

// ---------------- SAC Agent ----------------
class SACAgent {
public:
	const float log_2pi = std::log(2.0 * M_PI);
	
	SACAgent() :
		device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
		
		q1_net(STATE_DIM + ACTION_DIM, 256, 1),
		q2_net(STATE_DIM + ACTION_DIM, 256, 1),
		q1_target(STATE_DIM + ACTION_DIM, 256, 1),
		q2_target(STATE_DIM + ACTION_DIM, 256, 1),
		policy_net(STATE_DIM, 256, 2 * ACTION_DIM),
		log_alpha(torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(device))),
		
		opt_q1(q1_net->parameters(), torch::optim::AdamOptions(lr_q)),
		opt_q2(q2_net->parameters(), torch::optim::AdamOptions(lr_q)),
		opt_pi(policy_net->parameters(), torch::optim::AdamOptions(lr_pi)),
		opt_log_alpha(std::vector<torch::Tensor>{log_alpha}, torch::optim::AdamOptions(lr_alpha))
	{
		q1_net->to(device);
		q2_net->to(device);
		q1_target->to(device);
		q2_target->to(device);
		policy_net->to(device);
		
		torch::NoGradGuard no_grad;
		auto src1 = q1_net->named_parameters();
		auto tgt1 = q1_target->named_parameters();
		for (auto &p : src1) {
			const auto &name = p.key();
			tgt1[name].copy_(p.value());
		}
		auto src2 = q2_net->named_parameters();
		auto tgt2 = q2_target->named_parameters();
		for (auto &p : src2) {
			const auto &name = p.key();
			tgt2[name].copy_(p.value());
		}
		
//		torch::load(policy_net, "policy_net.pt");
	}
	
	std::vector<float> select_action(std::vector<float>& state) {
		auto state_tensor = torch::tensor(state).reshape({1, STATE_DIM}).to(device);
		torch::NoGradGuard no_grad;
		auto [action_tensor, _] = sample_action(state_tensor);
		auto action_cpu = action_tensor.to(torch::kCPU).item<float>(); //squeeze();
		std::vector<float> action = {action_cpu * 1.0f};
		
		if (ACTION_PARAM == 1) {
			std::cout << "action: " << action << std::endl;
		}
		return action;
	}
	
	void update(ReplayBuffer& memory) {
		if (memory.size() < BATCH_SIZE) return;
		
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
		
		auto alpha = log_alpha.exp();
		
		// --- Q target ---
		auto [next_action, next_log_pi] = sample_action(next_state_tensor);
		auto next_q1 = q1_target->forward(torch::cat({next_state_tensor, next_action}, 1));
		auto next_q2 = q2_target->forward(torch::cat({next_state_tensor, next_action}, 1));
		auto q_target_min = torch::min(next_q1, next_q2);
		auto q_target = reward_tensor + GAMMA * not_t_tensor * (q_target_min - ALPHA * next_log_pi); //
		q_target = q_target.detach();
		
		// --- Q updates ---
		auto q_input = torch::cat({state_tensor, action_tensor}, 1);
		auto q1 = q1_net->forward(q_input);
		auto q2 = q2_net->forward(q_input);
		
		auto loss_q1 = torch::mse_loss(q1, q_target);
		auto loss_q2 = torch::mse_loss(q2, q_target);
		
		if (LOSS_Q == 1) {
			std::cout << "loss_q1: " << loss_q1 << ", loss_q2: " << loss_q2 << std::endl;
		}
		
		opt_q1.zero_grad();
		opt_q2.zero_grad();
		loss_q1.backward();
		loss_q2.backward();
		torch::nn::utils::clip_grad_norm_(q1_net->parameters(), 1.0);
		torch::nn::utils::clip_grad_norm_(q2_net->parameters(), 1.0);
		opt_q1.step();
		opt_q2.step();
		
		// --- Policy update ---
		auto [a_pi, log_pi] = sample_action(state_tensor);
		auto q_input_pi = torch::cat({state_tensor, a_pi}, 1);
		auto q_min_pi = torch::min(q1_net->forward(q_input_pi), q2_net->forward(q_input_pi));
		if (Q_MIN == 1) {
			std::cout << "q_min_pi: " << q_min_pi << std::endl;
		}
		auto loss_pi = (q_min_pi - ALPHA * log_pi).mean();
		
		if (LOSS_POLICY == 1) {
			std::cout << "loss_pi: " << loss_pi << std::endl;
		}
		
		opt_pi.zero_grad();
		loss_pi.backward();
		torch::nn::utils::clip_grad_norm_(policy_net->parameters(), 1.0);
		opt_pi.step();
		
		// --- alpha ---
		auto alpha_loss = -(log_alpha * (log_pi.detach() - ACTION_DIM)).mean();
		
		opt_log_alpha.zero_grad();
		alpha_loss.backward();
		opt_log_alpha.step();
		
		// --- Q-target update ---
		update_target_net();
	}
	
	void save_net() {
		torch::save(policy_net, "policy_net.pt");
		std::cout << "Saved Policy_net" << std::endl;
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
	
	std::tuple<torch::Tensor, torch::Tensor> sample_action(torch::Tensor& state) {
		auto mean_lnsigma = policy_net->forward(state);
		auto mean = mean_lnsigma.slice(1, 0, ACTION_DIM);
		auto ln_sigma = torch::clamp(mean_lnsigma.slice(1, ACTION_DIM, 2 * ACTION_DIM), -20, 3);
		auto sigma_ = ln_sigma.exp();
		auto z = mean + sigma_ * torch::randn_like(mean);
		
		auto action = torch::tanh(z); //
		auto log_pi_z = -0.5 * (((z - mean) / sigma_).pow(2) + 2 * ln_sigma + log_2pi); //
		auto correction_term = torch::log(1.0 - action.pow(2) + 1e-6); // 加上小常數防止 log(0) //
		auto ln_pi = log_pi_z.sum(1, /*keepdim=*/true) - correction_term.sum(1, /*keepdim=*/true); //
		
//		auto ln_pi = ln_z.sum(1, /*keepdim=*/true); // log π(a|s)
		
//		action = mean; // test
		return std::make_tuple(action, ln_pi);
	}
	
	void update_target_net() {
		torch::NoGradGuard no_grad;
		for (auto &pair : q1_target->named_parameters()) {
			auto name = pair.key();                            // 取參數名稱（例如 "layer1.weight"）
			auto &tgt = pair.value();                          // 取得 target net 的參數 Tensor
			auto &src = q1_net->named_parameters()[name];      // 對應到 q1_net 的同名參數 Tensor
			tgt.copy_(src * (1.0 - TAU) + tgt * TAU);          // 按公式更新 target 參數
		}
		
		for (auto &pair : q2_target->named_parameters()) {
			auto name = pair.key();
			auto &tgt = pair.value();
			auto &src = q2_net->named_parameters()[name];
			tgt.copy_(src * (1.0 - TAU) + tgt * TAU);
		}
	}
};

float get_reward(std::vector<float>& state, std::vector<float>& next_state, float not_terminal) {
	float pole_reward = (1.5 - abs(next_state[2])) / 1.5;
	float cart_reward = 0;
	if (abs(next_state[2]) < 0.2) {
		cart_reward = (7.5 - abs(next_state[0] - 3)) / 7;
	}
	return pole_reward + cart_reward;
}

// ---------------- Main ----------------
int main() {
	// MuJoCo 初始化
	char error[1000];
	std::string xml_file = std::string(MUJOCO_MODEL_DIR) + "/cart_pole.xml";
	m = mj_loadXML(xml_file.c_str(), nullptr, error, 1000);
	if (!m) {
		std::cerr << "Failed to load XML: " << error << std::endl;
		return 1;
	}
	d = mj_makeData(m);
	
	// init GLFW, create window, make OpenGL context current // +++
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW" << "Failed to initialize GLFW" << std::endl;
		return 1;
	}
	GLFWwindow* window = glfwCreateWindow(1200, 700, "MuJoCo GUI", NULL, NULL);
	if (!window) {
		std::cerr << "Failed to create GLFW window" << std::endl;
		return 1;
	}
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
	
	// initialize visualization data structures // +++
	mjv_defaultCamera(&cam);
	mjv_defaultPerturb(&pert);
	mjv_defaultOption(&opt);
	mjr_defaultContext(&con);
	mjv_defaultScene(&scn);
	
	// create scene and context // +++
	mjv_makeScene(m, &scn, 1000);
	mjr_makeContext(m, &con, mjFONTSCALE_150);
	
	cam.type = mjCAMERA_FREE; // camera type (mjtCamera) // +++
	cam.lookat[0] = 0.0;
	cam.lookat[1] = 0.0; // lookat point
	cam.lookat[2] = 2.0;
	cam.distance = 15.0; // distance to lookat point or tracked body
	cam.azimuth = 90; // camera azimuth (deg)
	cam.elevation = -10; // camera elevation (deg)
	
	int cart_motor_id = mj_name2id(m, mjOBJ_ACTUATOR, "cart_motor");
	int cart_pos_id = mj_name2id(m, mjOBJ_SENSOR, "cart_pos");
	int cart_vel_id = mj_name2id(m, mjOBJ_SENSOR, "cart_vel");
	int pole_pos_id = mj_name2id(m, mjOBJ_SENSOR, "pole_pos");
	int pole_vel_id = mj_name2id(m, mjOBJ_SENSOR, "pole_vel");
	
	mjrRect viewport = {0, 0, 0, 0};
	glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
	mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
	mjr_render(viewport, &scn, &con);
	glfwSwapBuffers(window);
	glfwPollEvents();
	
	// 建立 agent
	SACAgent agent;
	ReplayBuffer memory;
	
	// 主訓練迴圈
	for (int episode = 0; episode < EPISODES; ++episode) {
		YAML::Node SAC_param = YAML::LoadFile("/home/yap/my_ws/src/mujoco_pkg/src/SAC.yaml");
		ACTION_PARAM = SAC_param["ACTION_PARAM"].as<int>();
		GLFW_SHOW = SAC_param["GLFW_SHOW"].as<int>();
		SAVE_NET = SAC_param["SAVE_NET"].as<int>();
		LOSS_POLICY = SAC_param["LOSS_POLICY"].as<int>();
		LOSS_Q = SAC_param["LOSS_Q"].as<int>();
		Q_MIN = SAC_param["Q_MIN"].as<int>();
		
		mj_resetData(m, d);
		mj_forward(m, d);
		
		float total_reward = 0;
		int step_count = 0;
		float not_terminal = 1;
		
		float cart_pos = d->sensordata[m->sensor_adr[cart_pos_id]];
		float cart_vel = d->sensordata[m->sensor_adr[cart_vel_id]];
		float pole_pos = d->sensordata[m->sensor_adr[pole_pos_id]];
		float pole_vel = d->sensordata[m->sensor_adr[pole_vel_id]];
		std::vector<float> state = {cart_pos, cart_vel, pole_pos, pole_vel};
		d->ctrl[cart_motor_id] = 0;
		
		while (not_terminal) {
			mjtNum simstart = d->time; // +++
			while (d->time - simstart < 1.0/20.0 && not_terminal) { // +++
			
			auto action = agent.select_action(state); //std::vector<float>
//			std::cout << "action: " << action << std::endl;
			float action_scalar = std::min(1000, 400 + 600 * (episode / (EPISODES - 200))); //200 -> 1000
			d->ctrl[cart_motor_id] = action[0] * action_scalar;
			
			mj_step(m, d); // 執行一個模擬步
			cart_pos = d->sensordata[m->sensor_adr[cart_pos_id]];
			cart_vel = d->sensordata[m->sensor_adr[cart_vel_id]];
			pole_pos = d->sensordata[m->sensor_adr[pole_pos_id]];
			pole_vel = d->sensordata[m->sensor_adr[pole_vel_id]];
			std::vector<float> next_state = {cart_pos, cart_vel, pole_pos, pole_vel};
			
			step_count = step_count + 1;
			if (step_count>4000 || cart_pos>4.5 || cart_pos<-4.5 || pole_pos<-1.5 || pole_pos>1.5) {
				not_terminal = 0;
			}
			
			float reward = get_reward(state, next_state, not_terminal);
			total_reward = total_reward + reward;
			
			memory.push(state, action, reward, next_state, not_terminal);
			
			agent.update(memory);
			
			state = next_state;
			} // dt < 1/60
			
			if (GLFW_SHOW == 1) {
				// get framebuffer viewport // +++
				mjrRect viewport = {0, 0, 0, 0};
				glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
				
				// update scene and render // +++
				mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
				mjr_render(viewport, &scn, &con);
				
				// swap OpenGL buffers (blocking call due to v-sync) // +++
				glfwSwapBuffers(window);
				
				// process pending GUI events, call GLFW callbacks // +++
				glfwPollEvents();
			}
		} // end episode loop
		
		std::cout << "Episode " << episode << ", Steps = " << step_count << ", Reward = " << total_reward << std::endl;
		
		if (SAVE_NET == 1) {
			agent.save_net();
		}
	} // end training loop
	
	mj_deleteData(d);
	mj_deleteModel(m);
	
	// close GLFW, free visualization storage // +++
	glfwTerminate();
	mjv_freeScene(&scn);
	mjr_freeContext(&con);
	return 0;
}


