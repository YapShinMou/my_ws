// 範例
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

// MuJoCo data structures
mjModel* m = nullptr;                  // MuJoCo model
mjData* d = nullptr;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context
mjvPerturb pert;

char c = 0; //盤輸入

//-------------------------------
// 非阻塞鍵盤輸入檢查 (類似 kbhit)
//-------------------------------
int kbhit(void) {
	struct termios oldt, newt;
	int ch;
	int oldf;
	
	tcgetattr(STDIN_FILENO, &oldt);
	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);
	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
	
	ch = getchar();
	
	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);
	
	if(ch != EOF) {
		ungetc(ch, stdin);
		return 1;
	}
	
	return 0;
}

//-------------------------------
// myfunc: 背景執行緒 (非阻塞讀鍵盤 + 等待)
//-------------------------------
void keyboard_input() {
	while (true) {
		if (kbhit()) {
			c = getchar();
			if (c == 'q') break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
}

int main() {
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
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW" << std::endl;
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
	int cart_tor_id = mj_name2id(m, mjOBJ_SENSOR, "cart_tor");
	int pole_tor_id = mj_name2id(m, mjOBJ_SENSOR, "pole_tor");
	
	float cart_tor;
	float pole_tor;
	
	// --- 啟動 myfunc 執行緒 ---
	std::thread thread_1(keyboard_input);
	
	// --- 主模擬迴圈 ---
	while (!glfwWindowShouldClose(window)) {
		mjtNum simstart = d->time;
		while (d->time - simstart < 1.0/60.0) {
			cart_tor = d->sensordata[m->sensor_adr[cart_tor_id]];
			if (c == 'a') {
				d->ctrl[cart_motor_id] = cart_tor - 50;
				std::cout << "  cart_tor: " << cart_tor - 50<< std::endl;
			} else if (c == 'd') {
				d->ctrl[cart_motor_id] = cart_tor + 50;
				std::cout << "  cart_tor: " << cart_tor + 50<< std::endl;
			}
			c = 's';
			
			mj_step(m, d);
		}
		
		// get framebuffer viewport
		mjrRect viewport = {0, 0, 0, 0};
		glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
		
		// update scene and render
		mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
		mjr_render(viewport, &scn, &con);
		
		// swap OpenGL buffers (blocking call due to v-sync)
		glfwSwapBuffers(window);
		
		// process pending GUI events, call GLFW callbacks
		glfwPollEvents();
	}
	
	thread_1.join();
	
	mj_deleteData(d);
	mj_deleteModel(m);
	
	// close GLFW, free visualization storage
	glfwTerminate();
	mjv_freeScene(&scn);
	mjr_freeContext(&con);
	
	return 0;
}



