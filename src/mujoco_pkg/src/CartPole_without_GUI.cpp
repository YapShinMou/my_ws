//範例程式
#include <mujoco/mujoco.h>
#include <iostream>

int main()
{
	// -----------------------------------
	// 載入模型與資料
	// -----------------------------------
	char error[1000];
	mjModel* m = mj_loadXML("/home/yap/my_ws/src/mujoco_pkg/models/cart_pole.xml", nullptr, error, 1000);
	if (!m) {
		std::cerr << "Failed to load XML: " << error << std::endl;
		return 1;
	}
	mjData* d = mj_makeData(m);
	
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
	for (int i=0; i<500; ++i)
	{
		// 模擬一步
		d->ctrl[cart_motor_id] = 500;
		mj_step(m, d);
		
		double cart_pos = d->sensordata[m->sensor_adr[cart_pos_id]];
		double cart_vel = d->sensordata[m->sensor_adr[cart_vel_id]];
		double pole_pos = d->sensordata[m->sensor_adr[pole_pos_id]];
		double pole_vel = d->sensordata[m->sensor_adr[pole_vel_id]];
		std::cout << "cart_pos=" << cart_pos << " pole_pos=" << pole_pos << std::endl;
	}
	
	// -----------------------------------
	// 清理
	// -----------------------------------
	mj_deleteData(d);
	mj_deleteModel(m);
	return 0;
}

