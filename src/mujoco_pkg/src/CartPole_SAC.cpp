// SAC_CartPole.cpp
#include <vector>
#include <deque>
#include <random>
// #include <chrono> ??????????????
#include <iostream>
#include <torch/torch.h>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <cmath> //???????????

// MuJoCo data structures
mjModel* m = nullptr;                  // MuJoCo model
mjData* d = nullptr;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context
mjvPerturb pert;

// ---------------- 超參數 ----------------
constexpr int STATE_DIM = 4;
constexpr int ACTION_DIM = 1;
constexpr int MEMORY_SIZE = 1000;
constexpr int BATCH_SIZE = 128;
constexpr int EPISODES = 1000;
constexpr int TARGET_UPDATE_INTERVAL = 10;

float GAMMA = 0.99;
float TAU = 0.005;
float ALPHA = 0.2;
float lr_v = 3e-4;
float lr_q = 3e-4;
float lr_pi = 3e-4;

// ---------------- 定義 MLP ----------------
struct MLPImpl : torch::nn::Module {
    torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr};
    MLPImpl(int in_dim, int hidden, int out_dim) {
        l1 = register_module("l1", torch::nn::Linear(in_dim, hidden));
        l2 = register_module("l2", torch::nn::Linear(hidden, hidden));
        l3 = register_module("l3", torch::nn::Linear(hidden, out_dim));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(l1->forward(x));
        x = torch::relu(l2->forward(x));
        return l3->forward(x);
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
	SACAgent() :
		device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
		v_net(STATE_DIM, 256, 1),
		v_target(STATE_DIM, 256, 1),
		q1_net(STATE_DIM + ACTION_DIM, 256, 1),
		q2_net(STATE_DIM + ACTION_DIM, 256, 1),
		policy_net(STATE_DIM, 256, 2 * ACTION_DIM),
		opt_v(v_net->parameters(), torch::optim::AdamOptions(lr_v)),
		opt_q1(q1_net->parameters(), torch::optim::AdamOptions(lr_q)),
		opt_q2(q2_net->parameters(), torch::optim::AdamOptions(lr_q)),
		opt_pi(policy_net->parameters(), torch::optim::AdamOptions(lr_pi))
	{
		v_net->to(device);
		v_target->to(device);
		q1_net->to(device);
		q2_net->to(device);
		policy_net->to(device);
		
//		torch::load(policy_net, "policy_net.pt"); //
		
		torch::NoGradGuard no_grad;
		auto src_params = v_net->named_parameters();
		auto tgt_params = v_target->named_parameters();
		for (auto &pair : src_params) {
			auto name = pair.key();
			auto &src_tensor = pair.value();
			auto &tgt_tensor = tgt_params[name];
			tgt_tensor.copy_(src_tensor);
		}
		for (auto &p : v_target->parameters()) {
			p.set_requires_grad(false);
		}
	}
	
	std::vector<float> select_action(const std::vector<float>& state){
		auto state_tensor = torch::tensor(state).reshape({1, STATE_DIM}).to(device);
		torch::NoGradGuard no_grad;
		auto action_tensor = sample_action(state_tensor);
		auto action_cpu = action_tensor.to(torch::kCPU).squeeze();
		std::vector<float> action(action_cpu.data_ptr<float>(),
                                  action_cpu.data_ptr<float>() + action_cpu.numel());
		return action;
	}
	
	void update(ReplayBuffer &memory) {
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
		
		// --- Q target ---
		auto next_action = sample_action(next_state_tensor);
		auto next_q1 = q1_net->forward(torch::cat({next_state_tensor, next_action}, 1));
		auto next_q2 = q2_net->forward(torch::cat({next_state_tensor, next_action}, 1));
		auto q_target_min = torch::min(next_q1, next_q2);
		auto q_target = reward_tensor + GAMMA * not_t_tensor * (q_target_min - ALPHA * ln_pi.detach());
		
		// --- Q updates ---
		auto q_input = torch::cat({state_tensor, action_tensor}, 1);
		auto q1 = q1_net->forward(q_input);
		auto q2 = q2_net->forward(q_input);
		
		auto loss_q1 = torch::mse_loss(q1, q_target.detach());
		auto loss_q2 = torch::mse_loss(q2, q_target.detach());
		
		opt_q1.zero_grad(); loss_q1.backward(); opt_q1.step();
		opt_q2.zero_grad(); loss_q2.backward(); opt_q2.step();
		
		// --- Value update ---
		auto a_pi = sample_action(state_tensor);
		auto q_input_pi = torch::cat({state_tensor, a_pi}, 1);
		auto q_min = torch::min(q1_net->forward(q_input_pi), q2_net->forward(q_input_pi));
		auto v_target_now = (q_min.detach() - ALPHA * ln_pi.detach());
		auto loss_v = torch::mse_loss(v_net->forward(state_tensor), v_target_now);
		
		opt_v.zero_grad();
		loss_v.backward();
		opt_v.step();
		
		// --- Policy update ---
		auto q_min_pi = torch::min(q1_net->forward(q_input_pi), q2_net->forward(q_input_pi));
		auto loss_pi = (ALPHA * ln_pi - q_min_pi).mean();
		
		opt_pi.zero_grad();
		loss_pi.backward();
		opt_pi.step();
		
		// --- Soft update target ---
		torch::NoGradGuard no_grad;
		auto src_params = v_net->named_parameters();
		auto tgt_params = v_target->named_parameters();
		for (auto &pair : tgt_params) {
			const auto &name = pair.key();
			auto &tgt = pair.value();
			auto &src = src_params[name];
			tgt.copy_(tgt * (1.0 - TAU) + src * TAU);
		}
	}
	
	void save_net() {
		torch::save(policy_net, "policy_net.pt");
	}
	
private:
	torch::Device device;
	
	MLP v_net{nullptr};
	MLP v_target{nullptr};
	MLP q1_net{nullptr};
	MLP q2_net{nullptr};
	MLP policy_net{nullptr};
	
	torch::optim::Adam opt_v;
	torch::optim::Adam opt_q1;
	torch::optim::Adam opt_q2;
	torch::optim::Adam opt_pi;
	
	torch::Tensor ln_pi;
	
//	torch::Tensor sample_action(torch::Tensor state, torch::Tensor &logp) {
//		auto mean_logstd = policy_net->forward(state);
//		auto mean = mean_logstd.slice(1, 0, ACTION_DIM);
//		auto log_std = mean_logstd.slice(1, ACTION_DIM, 2 * ACTION_DIM).clamp(-20, 2);
//		auto std = log_std.exp();
//		auto eps = torch::randn_like(mean);
//		auto a = mean + std * eps;
//		auto action = torch::tanh(a);
		
		// log π(a|s)
//		logp = -0.5 * ((eps.pow(2) + 2 * log_std + std::log(2 * M_PI)).sum(1, true));
//		logp -= torch::log(1 - action.pow(2) + 1e-6).sum(1, true);
		
//		return action;
//	}
	
	torch::Tensor sample_action(torch::Tensor state) {
		auto mean_logstd = policy_net->forward(state);
		auto mean = mean_logstd.slice(1, 0, ACTION_DIM);
		auto ln_sigma = mean_logstd.slice(1, ACTION_DIM, 2 * ACTION_DIM).clamp(-20, 2);
		auto sigma_ = ln_sigma.exp();
		auto eps = torch::randn_like(mean);
		auto a = mean + sigma_ * eps;
		auto action = torch::tanh(a);
		
		// log π(a|s)
		ln_pi = -0.5 * ((a - mean)*(a - mean) / (sigma_*sigma_) + 2*ln_sigma + torch::log(torch::tensor(2 * M_PI, torch::TensorOptions().device(device))));
		ln_pi = ln_pi.sum(1, true);
		ln_pi = ln_pi - (2 * (torch::log(torch::tensor(2.0)) - a - torch::nn::functional::softplus(-2 * a))).sum(1, true);
		
		return action;
	}
};

float get_reward(const std::vector<float>& state, const std::vector<float>& next_state, float not_terminal) {
	if (next_state[2] > -0.5 && next_state[2] < 0.5) {
		return 1 - 2*abs(next_state[2]);
	} else {
		return 0;
	}
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
	
	// init GLFW, create window, make OpenGL context current
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
	
	// initialize visualization data structures
	mjv_defaultCamera(&cam);
	mjv_defaultPerturb(&pert);
	mjv_defaultOption(&opt);
	mjr_defaultContext(&con);
	mjv_defaultScene(&scn);
	
	// create scene and context
	mjv_makeScene(m, &scn, 1000);
	mjr_makeContext(m, &con, mjFONTSCALE_150);
	
	cam.type = mjCAMERA_FREE; // camera type (mjtCamera)
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
	
	// 建立 agent
	SACAgent agent;
	ReplayBuffer memory;
	
	// 主訓練迴圈
	for (int episode = 0; episode < EPISODES; ++episode) {
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
			mjtNum simstart = d->time;
			while (d->time - simstart < 1.0/60) {
			
			auto action = agent.select_action(state); //std::vector<float>
			d->ctrl[cart_motor_id] = action[0] * 1000;
			
			mj_step(m, d); // 執行一個模擬步
			cart_pos = d->sensordata[m->sensor_adr[cart_pos_id]];
			cart_vel = d->sensordata[m->sensor_adr[cart_vel_id]];
			pole_pos = d->sensordata[m->sensor_adr[pole_pos_id]];
			pole_vel = d->sensordata[m->sensor_adr[pole_vel_id]];
			std::vector<float> next_state = {cart_pos, cart_vel, pole_pos, pole_vel};
			
			step_count = step_count + 1;
			if (step_count>2000 || cart_pos>4.5 || cart_pos<-4.5 || pole_pos<-0.5 || pole_pos>0.5) not_terminal = 0;
			
			float reward = get_reward(state, next_state, not_terminal);
			total_reward = total_reward + reward;
			
			memory.push(state, action, reward, next_state, not_terminal);
			
			agent.update(memory);
			
			state = next_state;
			} // dt < 1/60
			
			// get framebuffer viewport
			mjrRect viewport = {0, 0, 0, 0};
			glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
			
			// update scene and render
			mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
			mjr_render(viewport, &scn, &con);
			
			// swap OpenGL buffers (blocking call due to v-sync)
			glfwSwapBuffers(window);
			
			// process pending GUI events, call GLFW callbacks
			glfwPollEvents();
		} // end episode loop
		std::cout << "Episode " << episode << ", total steps = " << step_count << std::endl;
	} // end training loop
	
	agent.save_net();
	
	mj_deleteData(d);
	mj_deleteModel(m);
	
	// close GLFW, free visualization storage
	glfwTerminate();
	mjv_freeScene(&scn);
	mjr_freeContext(&con);
	return 0;
}



