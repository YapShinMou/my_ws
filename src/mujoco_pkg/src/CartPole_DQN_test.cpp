#include <thread>
#include <mutex>
#include <vector>
#include <deque>
#include <random>
#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

// ==================== 超參數 ====================
const int input_dim = 4;
const int output_dim = 7;
//constexpr int BATCH_SIZE = 64;
//constexpr int MEMORY_SIZE = 1000;
//constexpr float LR = 1e-3f; //learning rate

//constexpr int EPISODES = 400;
//constexpr float GAMMA = 0.9;
//constexpr float EPS_START = 1.0f;
//constexpr float EPS_END = 0.01f;
//constexpr float EPS_DECAY = 100000.0f;
//constexpr int TARGET_UPDATE = 10;

// 全域變數（參考官方）
mjvCamera cam;
mjvOption opt;
mjvScene scn;
mjrContext con;

// ==================== DQN Implement ====================
struct NetImpl : torch::nn::Module {
	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
	
	NetImpl() {
		fc1 = register_module("fc1", torch::nn::Linear(input_dim, 128));
		fc2 = register_module("fc2", torch::nn::Linear(128, 128));
		fc3 = register_module("fc3", torch::nn::Linear(128, output_dim));
	}
	
	torch::Tensor forward(torch::Tensor x) {
		x = torch::leaky_relu(fc1->forward(x));
		x = torch::leaky_relu(fc2->forward(x));
		x = fc3->forward(x);
		return x;
	}
};
TORCH_MODULE(Net);

int select_action(Net& policy_net, const std::vector<double>& state, float epsilon, const torch::Device& device) {
//	std::uniform_real_distribution<float> maxQ_or_random(0.0f,1.0f);
//	std::mt19937 rng{std::random_device{}()};
//	if (maxQ_or_random(rng) < epsilon) {
//		std::uniform_int_distribution<int> random_action(0, output_dim-1);
//		return random_action(rng);
//	} else {
		auto x = torch::tensor(state, torch::kFloat32).reshape({1, input_dim}).to(device);
		auto output = policy_net->forward(x);
		auto indx = output.argmax(1).item<int64_t>();
		return static_cast<int>(indx);
//	}
}

// ---------- main ----------
int main()
{
	// -----------------------------------
	// 載入模型與資料
	// -----------------------------------
	char error[1000];
	std::string xml_file = std::string(MUJOCO_MODEL_DIR) + "/cart_pole.xml";
	mjModel* m = mj_loadXML(xml_file.c_str(), nullptr, error, 1000);
	if (!m) {
		std::cerr << "Failed to load XML: " << error << std::endl;
		return 1;
	}
	mjData* d = mj_makeData(m);
	
	// -----------------------------------
	// 初始化 GLFW
	// -----------------------------------
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
	glfwSwapInterval(1);  // 垂直同步
	
	// -----------------------------------
	// 初始化 MuJoCo 可視化
	// -----------------------------------
	mjv_defaultCamera(&cam);
	mjv_defaultOption(&opt);
	mjv_defaultScene(&scn);
	mjr_defaultContext(&con);
	
	mjv_makeScene(m, &scn, 1000);
	mjr_makeContext(m, &con, mjFONTSCALE_150);
	
	cam.type = mjCAMERA_FREE;  // 使用自由相機模式（避免跟模型綁定）
	cam.lookat[0] = 0.0;
	cam.lookat[1] = 0.0;
	cam.lookat[2] = 2.0;   // 高度視角，看起來會稍微往下看
	cam.distance = 15.0;
	cam.azimuth = 90;   // 左右旋轉角
	cam.elevation = -10; // 上下旋轉角
	
	// -----------------------------------
	// 控制與感測器 ID
	// -----------------------------------
	int cart_motor_id = mj_name2id(m, mjOBJ_ACTUATOR, "cart_motor");
	int cart_pos_id = mj_name2id(m, mjOBJ_SENSOR, "cart_pos");
	int cart_vel_id = mj_name2id(m, mjOBJ_SENSOR, "cart_vel");
	int pole_pos_id = mj_name2id(m, mjOBJ_SENSOR, "pole_pos");
	int pole_vel_id = mj_name2id(m, mjOBJ_SENSOR, "pole_vel");
	
	double cart_pos;
	double cart_vel;
	double pole_pos;
	double pole_vel;
	
	// -----------------------------------
	// 初始神經網路
	// -----------------------------------
	torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	std::cout << "Using device: " << (device.is_cuda() ? "CUDA (GPU)" : "CPU") << std::endl;
	
	auto policy_net = Net();
//	auto target_net = Net();
	policy_net->to(device);
//	target_net->to(device);
//	torch::optim::Adam optimizer(policy_net->parameters(), torch::optim::AdamOptions(LR)); // optimizer 要追蹤梯度，optimizer 放在train()裡面會把訓練重置掉
//	torch::save(policy_net, "tmp.pt");
//	torch::load(target_net, "tmp.pt");
//	target_net->eval(); //設定成推論模式
	torch::load(policy_net, "policy_net.pt");
	
//	ReplayBuffer memory;
	
//	int steps_done = 0;
	
	// -----------------------------------
	// 主迴圈
	// -----------------------------------
	for (int episode = 0; episode < 10 ; ++episode)
	{
		mj_resetData(m, d);
		mj_forward(m, d);
		
//		float epsilon = EPS_START;
//		float total_reward = 0.0f;
		bool done = false;
		int step_count = 0;
		
		cart_pos = d->sensordata[m->sensor_adr[cart_pos_id]];
		cart_vel = d->sensordata[m->sensor_adr[cart_vel_id]];
		pole_pos = d->sensordata[m->sensor_adr[pole_pos_id]];
		pole_vel = d->sensordata[m->sensor_adr[pole_vel_id]];
		std::vector<double> state = {cart_pos, cart_vel, pole_pos, pole_vel};
		d->ctrl[cart_motor_id] = 0;
		
		// episode loop
		while (!done) {
			mjtNum simstart = d->time;
			while (d->time - simstart < 1.0/60) {
//				std::cout << "time: " << d->time << std::endl;
				
				// compute epsilon
//				epsilon = EPS_END + (EPS_START - EPS_END) * std::exp(-1.0f * float(steps_done) / EPS_DECAY);
//				steps_done = steps_done + 1;
				
				// choose action
				int action = select_action(policy_net, state, 0, device);
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
//				std::vector<double> next_state = {cart_pos, cart_vel, pole_pos, pole_vel};
				
//				float reward = get_reward(state, next_state, done);
//				total_reward += reward;
				
				step_count = step_count + 1;
				if (step_count>3000 || cart_pos>4.5 || cart_pos<-4.5 || pole_pos<-0.8 || pole_pos>0.8) done = true;
				
				// store to memory
//				memory.push(state, action, reward, next_state, done);
				
				// update state
//				state = next_state;
				
				// train if enough samples
//				if (memory.size() >= BATCH_SIZE) {
//					std::vector<ReplayBuffer::Experience> batch = memory.sample_shuffle_indices();
//					train(policy_net, target_net, optimizer, batch, device);
//				}
			
			}
			
			mjrRect viewport = {0, 0, 0, 0};
			glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
			mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
			mjr_render(viewport, &scn, &con);
			glfwSwapBuffers(window);
			glfwPollEvents();
		} // end episode loop
		
//		if (episode % TARGET_UPDATE == 0) {
//			torch::save(policy_net, "tmp.pt");
//			torch::load(target_net, "tmp.pt");
			std::cout << "Episode " << episode << " update target net" << std::endl;
//			std::cout << "epsilon: " << epsilon << std::endl;
//		}
		std::cout << "total time-steps: " << step_count << std::endl;
	} // end training loop
	
	// save policy
//	try { //嘗試執行可能會失敗的操作
//		torch::save(policy_net, "policy_net.pt");
//		std::cout << "Saved policy_net.pt " << std::endl;
//	} catch (const std::exception &e) { //如果失敗，讀取錯誤訊息
//		std::cout << "Failed to save model" << e.what() << std::endl;
//	}
	
	// -----------------------------------
	// 清理
	// -----------------------------------
	mj_deleteData(d);
	mj_deleteModel(m);
	
	glfwTerminate();
	mjv_freeScene(&scn);
	mjr_freeContext(&con);
	return 0;
}
	
	
	
	
	


