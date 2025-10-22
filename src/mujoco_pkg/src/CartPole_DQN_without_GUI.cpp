// 範例程式
//#include <string>
//#include <thread>
#include <mutex>
#include <vector>
#include <deque>
#include <random>
#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <mujoco/mujoco.h>
//#include <GLFW/glfw3.h>

// MuJoCo data structures
mjModel* m = nullptr;                  // MuJoCo model
mjData* d = nullptr;                   // MuJoCo data
//mjvCamera cam;                      // abstract camera
//mjvOption opt;                      // visualization options
//mjvScene scn;                       // abstract scene
//mjrContext con;                     // custom GPU context
//mjvPerturb pert;

// ==================== 超參數 ====================
const int input_dim = 4;
const int output_dim = 7;
constexpr int MEMORY_SIZE = 1000;
constexpr int BATCH_SIZE = 128;
constexpr float LR = 1e-4f; //learning rate

constexpr int EPISODES = 1500;
constexpr float GAMMA = 0.99;
constexpr float EPS_START = 1.0f;
constexpr float EPS_END = 0.01f;
constexpr float EPS_DECAY = 200000.0f;
constexpr int TARGET_UPDATE = 10;

// ==================== DQN Implement ====================
struct NetImpl : torch::nn::Module {
	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
	
	NetImpl() {
		fc1 = register_module("fc1", torch::nn::Linear(input_dim, 256));
		fc2 = register_module("fc2", torch::nn::Linear(256, 256));
		fc3 = register_module("fc3", torch::nn::Linear(256, output_dim));
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
	
	void push(const std::vector<float>& state, int action, float reward, const std::vector<float>& next_state, bool done) {
		if (buffer_.size() >= MEMORY_SIZE) buffer_.pop_front();
		buffer_.push_back({state, action, reward, next_state, done});
	}
	
	std::vector<Experience> sample() {
		std::vector<Experience> batch; //宣告一個名稱叫做 batch 的變數，型態為 std::vector<Experience>
		std::uniform_int_distribution<int> dist(0, buffer_.size() - 1); //dist(rng) 為0到buffer_.size() - 1隨機整數
		for (int i=0; i<BATCH_SIZE; i++) {
			batch.push_back(buffer_[dist(rng)]);
		}
		return batch;
	}
	
	std::vector<Experience> sample_shuffle_indices() {
		std::vector<Experience> batch;
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

float get_reward(const std::vector<float>& state, const std::vector<float>& next_state, bool done) {
	if (next_state[2] > -0.5 && next_state[2] < 0.5) {
		return 1 - 2*abs(next_state[2]);
	} else {
		return 0;
	}
}

int select_action(Net& policy_net, const std::vector<float>& state, float epsilon, const torch::Device& device) {
	std::uniform_real_distribution<float> maxQ_or_random(0.0f,1.0f);
	std::mt19937 rng{std::random_device{}()};
	if (maxQ_or_random(rng) < epsilon) {
		std::uniform_int_distribution<int> random_action(0, output_dim-1);
		return random_action(rng);
	} else {
		auto x = torch::tensor(state, torch::kFloat32).reshape({1, input_dim}).to(device);
		auto output = policy_net->forward(x);
		auto indx = output.argmax(1).item<int64_t>();
		return static_cast<int>(indx);
	}
}

void train(Net& policy_net, 
				Net& target_net, 
				torch::optim::Optimizer& optimizer, 
				const std::vector<ReplayBuffer::Experience>& batch, 
				const torch::Device& device)
{
	int batch_size = batch.size();
	torch::nn::MSELoss loss_;
	
	std::vector<float> all_state, all_next_state;
	std::vector<int> all_action;
	std::vector<float> all_reward;
	std::vector<float> all_done;
	
	all_state.reserve(batch_size * input_dim);
	all_action.reserve(batch_size);
	all_reward.reserve(batch_size);
	all_next_state.reserve(batch_size * input_dim);
	all_done.reserve(batch_size);
	
	for (const auto& e : batch) {
		for (int i = 0; i < input_dim; ++i) {
			all_state.push_back(static_cast<float>(e.state[i]));
			all_next_state.push_back(static_cast<float>(e.next_state[i]));
		}
		all_action.push_back(e.action);
		all_reward.push_back(e.reward);
		all_done.push_back(e.done ? 0.0f : 1.0f);  // done=true -> 0
	}
	
	auto state_tensor = torch::tensor(all_state).reshape({batch_size, input_dim}).to(device);
	auto action_tensor = torch::tensor(all_action, torch::kInt64).to(device);
	auto reward_tensor = torch::tensor(all_reward).to(device);
	auto next_state_tensor = torch::tensor(all_next_state).reshape({batch_size, input_dim}).to(device);
	auto done_tensor = torch::tensor(all_done).to(device);
	
	auto q_values_all = policy_net->forward(state_tensor); // 取出各個狀態下的所有Q值
	auto q_values = q_values_all.gather(1, action_tensor.unsqueeze(1)).squeeze(1); // 只留下action相對應的Q(s, a)
	
	auto next_q_all = target_net->forward(next_state_tensor);
	auto next_q_values = std::get<0>(next_q_all.max(1)); // max(Q)
	
	auto expected_q = reward_tensor + GAMMA * next_q_values * done_tensor;
	
	auto loss = loss_(q_values, expected_q.detach());
	
	// ------ Backpropagate ------
	optimizer.zero_grad(); // 清空上一輪梯度
	loss.backward();       // 反向傳播
	torch::nn::utils::clip_grad_norm_(policy_net->parameters(), 1.0); // 防止梯度爆炸
	optimizer.step();      // 更新權重
}

// ---------- main ----------
int main()
{
	// ... load model and data
	char error[1000];
	std::string xml_file = std::string(MUJOCO_MODEL_DIR) + "/cart_pole.xml";
	m = mj_loadXML(xml_file.c_str(), nullptr, error, 1000);
	if (!m) {
		std::cerr << "Failed to load XML: " << error << std::endl;
		return 1;
	}
	d = mj_makeData(m);
	
	// init GLFW, create window, make OpenGL context current
//	if (!glfwInit()) {
//		std::cerr << "Failed to initialize GLFW" << "Failed to initialize GLFW" << std::endl;
//		return 1;
//	}
//	GLFWwindow* window = glfwCreateWindow(1200, 700, "MuJoCo GUI", NULL, NULL);
//	if (!window) {
//		std::cerr << "Failed to create GLFW window" << std::endl;
//		return 1;
//	}
//	glfwMakeContextCurrent(window);
//	glfwSwapInterval(1);
	
	// initialize visualization data structures
//	mjv_defaultCamera(&cam);
//	mjv_defaultPerturb(&pert);
//	mjv_defaultOption(&opt);
//	mjr_defaultContext(&con);
//	mjv_defaultScene(&scn);
	
	// create scene and context
//	mjv_makeScene(m, &scn, 1000);
//	mjr_makeContext(m, &con, mjFONTSCALE_150);
	
//	cam.type = mjCAMERA_FREE; // camera type (mjtCamera)
//	cam.lookat[0] = 0.0;
//	cam.lookat[1] = 0.0; // lookat point
//	cam.lookat[2] = 2.0;
//	cam.distance = 15.0; // distance to lookat point or tracked body
//	cam.azimuth = 90; // camera azimuth (deg)
//	cam.elevation = -10; // camera elevation (deg)
	
	int cart_motor_id = mj_name2id(m, mjOBJ_ACTUATOR, "cart_motor");
	int cart_pos_id = mj_name2id(m, mjOBJ_SENSOR, "cart_pos");
	int cart_vel_id = mj_name2id(m, mjOBJ_SENSOR, "cart_vel");
	int pole_pos_id = mj_name2id(m, mjOBJ_SENSOR, "pole_pos");
	int pole_vel_id = mj_name2id(m, mjOBJ_SENSOR, "pole_vel");
	
	float cart_pos;
	float cart_vel;
	float pole_pos;
	float pole_vel;
	
	// -----------------------------------
	// 初始神經網路
	// -----------------------------------
	torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	std::cout << "Using device: " << (device.is_cuda() ? "CUDA (GPU)" : "CPU") << std::endl;
	
	auto policy_net = Net();
	auto target_net = Net();
	policy_net->to(device);
	target_net->to(device);
	torch::optim::Adam optimizer(policy_net->parameters(), torch::optim::AdamOptions(LR)); // optimizer 要追蹤梯度，optimizer 放在train()裡面會把訓練重置掉
	torch::save(policy_net, "tmp.pt");
	torch::load(target_net, "tmp.pt");
	target_net->eval(); //設定成推論模式
	
	ReplayBuffer memory;
	
	int steps_done = 0;
	
	// -----------------------------------
	// 主迴圈
	// -----------------------------------
	for (int episode = 0; episode < EPISODES ; ++episode)
	{
		mj_resetData(m, d);
		mj_forward(m, d);
		
		float epsilon = EPS_START;
		float total_reward = 0.0f;
		bool done = false;
		int step_count = 0;
		
		cart_pos = d->sensordata[m->sensor_adr[cart_pos_id]];
		cart_vel = d->sensordata[m->sensor_adr[cart_vel_id]];
		pole_pos = d->sensordata[m->sensor_adr[pole_pos_id]];
		pole_vel = d->sensordata[m->sensor_adr[pole_vel_id]];
		std::vector<float> state = {cart_pos, cart_vel, pole_pos, pole_vel};
		d->ctrl[cart_motor_id] = 0;
		
		// episode loop
		while (!done) {
//			mjtNum simstart = d->time;
//			while (d->time - simstart < 1.0/60) {
//				std::cout << "time: " << d->time << std::endl;
				
				// compute epsilon
				epsilon = EPS_END + (EPS_START - EPS_END) * std::exp(-1.0f * float(steps_done) / EPS_DECAY);
				steps_done = steps_done + 1;
				
				// choose action
				int action = select_action(policy_net, state, epsilon, device);
				if (action == 0) {
					d->ctrl[cart_motor_id] = -900;
				} else if (action == 1) {
					d->ctrl[cart_motor_id] = -600;
				} else if (action == 2) {
					d->ctrl[cart_motor_id] = -300;
				} else if (action == 3) {
					d->ctrl[cart_motor_id] = 0;
				} else if (action == 4) {
					d->ctrl[cart_motor_id] = 300;
				} else if (action == 5) {
					d->ctrl[cart_motor_id] = 600;
				} else { 
					d->ctrl[cart_motor_id] = 900;
				}
				
				// 模擬一步
				mj_step(m, d);
				cart_pos = d->sensordata[m->sensor_adr[cart_pos_id]];
				cart_vel = d->sensordata[m->sensor_adr[cart_vel_id]];
				pole_pos = d->sensordata[m->sensor_adr[pole_pos_id]];
				pole_vel = d->sensordata[m->sensor_adr[pole_vel_id]];
				std::vector<float> next_state = {cart_pos, cart_vel, pole_pos, pole_vel};
				
				float reward = get_reward(state, next_state, done);
				total_reward += reward;
				
				step_count = step_count + 1;
				if (step_count>4000 || cart_pos>4.5 || cart_pos<-4.5 || pole_pos<-0.5 || pole_pos>0.5) done = true;
				
				// store to memory
				memory.push(state, action, reward, next_state, done);
				
				// update state
				state = next_state;
				
				// train if enough samples
				if (memory.size() >= BATCH_SIZE) {
					std::vector<ReplayBuffer::Experience> batch = memory.sample_shuffle_indices();
					train(policy_net, target_net, optimizer, batch, device);
				}
			
//			}
			
			// get framebuffer viewport
//			mjrRect viewport = {0, 0, 0, 0};
//			glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
			
			// update scene and render
//			mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
//			mjr_render(viewport, &scn, &con);
			
			// swap OpenGL buffers (blocking call due to v-sync)
//			glfwSwapBuffers(window);
			
			// process pending GUI events, call GLFW callbacks
//			glfwPollEvents();
		} // end episode loop
		
		if (episode % TARGET_UPDATE == 0) {
			torch::save(policy_net, "tmp.pt");
			torch::load(target_net, "tmp.pt");
			std::cout << "Episode " << episode << " update target net" << std::endl;
			std::cout << "epsilon: " << epsilon << std::endl;
		}
		
		std::cout << "total time-steps: " << step_count << std::endl;
	} // end training loop
	
	// save policy
	try { //嘗試執行可能會失敗的操作
		torch::save(policy_net, "policy_net.pt");
		std::cout << "Saved policy_net.pt " << std::endl;
	} catch (const std::exception &e) { //如果失敗，讀取錯誤訊息
		std::cout << "Failed to save model" << e.what() << std::endl;
	}
	
	// -----------------------------------
	// 清理
	// -----------------------------------
	mj_deleteData(d);
	mj_deleteModel(m);
	
// close GLFW, free visualization storage
//	glfwTerminate();
//	mjv_freeScene(&scn);
//	mjr_freeContext(&con);
	return 0;
}
	
	
	
	
	


