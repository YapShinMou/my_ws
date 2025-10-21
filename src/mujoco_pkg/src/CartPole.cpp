#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

mjvCamera cam;
mjvOption opt;
mjvScene scn;
mjrContext con;

char c = 0;

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
void myfunc() {
	while (true) {
		if (kbhit()) {
			c = getchar();
//			std::cout << "你按了: " << c << std::endl;
			if (c == 'q') break;
		}
		// 加入等待，避免 CPU 100%
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
}

int main() {
	// --- 載入模型 ---
	char error[1000];
	std::string xml_file = std::string(MUJOCO_MODEL_DIR) + "/cart_pole.xml";
	mjModel* m = mj_loadXML(xml_file.c_str(), nullptr, error, 1000);
	if (!m) {
		std::cerr << "Failed to load XML: " << error << std::endl;
		return 1;
	}
	mjData* d = mj_makeData(m);
	
	// --- 初始化 GLFW ---
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW" << std::endl;
		return 1;
	}
	GLFWwindow* window = glfwCreateWindow(1200, 700, "MuJoCo GUI", NULL, NULL);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
	
	// --- 初始化 MuJoCo ---
	mjv_defaultCamera(&cam);
	mjv_defaultOption(&opt);
	mjv_defaultScene(&scn);
	mjr_defaultContext(&con);
	mjv_makeScene(m, &scn, 1000);
	mjr_makeContext(m, &con, mjFONTSCALE_150);
	
	cam.type = mjCAMERA_FREE;
	cam.lookat[2] = 2.0;
	cam.distance = 15.0;
	cam.azimuth = 90;
	cam.elevation = -10;
	
	int cart_motor_id = mj_name2id(m, mjOBJ_ACTUATOR, "cart_motor");
	int cart_tor_id = mj_name2id(m, mjOBJ_SENSOR, "cart_tor");
	int pole_tor_id = mj_name2id(m, mjOBJ_SENSOR, "pole_tor");
	
	float cart_tor;
	float pole_tor;
	
	// --- 啟動 myfunc 執行緒 ---
	std::thread t1(myfunc);
	
	// --- 主模擬迴圈 ---
	while (!glfwWindowShouldClose(window)) {
		cart_tor = d->sensordata[m->sensor_adr[cart_tor_id]];
		
		
		mjtNum simstart = d->time;
		while (d->time - simstart < 1.0/60.0) {
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
		
		mjrRect viewport = {0, 0, 0, 0};
		glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
		mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
		mjr_render(viewport, &scn, &con);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	
	t1.join();
	
	mj_deleteData(d);
	mj_deleteModel(m);
	mjv_freeScene(&scn);
	mjr_freeContext(&con);
	glfwTerminate();
	return 0;
}



