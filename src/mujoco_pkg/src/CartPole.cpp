//範例程式
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <iostream>

// 全域變數（參考官方）
mjvCamera cam;
mjvOption opt;
mjvScene scn;
mjrContext con;

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
	
	// -----------------------------------
	// 主迴圈
	// -----------------------------------
	while( !glfwWindowShouldClose(window) )
	{
		d->ctrl[cart_motor_id] = 100;
		
		mjtNum simstart = d->time;
		while (d->time - simstart < 1.0/60.0) {
			// 模擬一步
			mj_step(m, d);
		}
		
		double cart_pos = d->sensordata[m->sensor_adr[cart_pos_id]];
		double cart_vel = d->sensordata[m->sensor_adr[cart_vel_id]];
		double pole_pos = d->sensordata[m->sensor_adr[pole_pos_id]];
		double pole_vel = d->sensordata[m->sensor_adr[pole_vel_id]];
//		std::cout << "cart_pos=" << cart_pos << " pole_pos=" << pole_pos << std::endl;
		
		mjrRect viewport = {0, 0, 0, 0};
		glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
		
		// 更新畫面
		mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
		mjr_render(viewport, &scn, &con);
		
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	
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

