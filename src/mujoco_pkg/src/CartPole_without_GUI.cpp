//範例程式
#include <mujoco/mujoco.h>
#include <iostream>

// MuJoCo data structures
mjModel* m = nullptr;                  // MuJoCo model
mjData* d = nullptr;                   // MuJoCo data

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
	
	int cart_motor_id = mj_name2id(m, mjOBJ_ACTUATOR, "cart_motor");
	int cart_pos_id = mj_name2id(m, mjOBJ_SENSOR, "cart_pos");
	int cart_vel_id = mj_name2id(m, mjOBJ_SENSOR, "cart_vel");
	int pole_pos_id = mj_name2id(m, mjOBJ_SENSOR, "pole_pos");
	int pole_vel_id = mj_name2id(m, mjOBJ_SENSOR, "pole_vel");
	
	d->ctrl[cart_motor_id] = 500;
	
	for (int i=0; i<10000; ++i)
	{
		if (i > 5000) {
			d->ctrl[cart_motor_id] = -900;
		}
		
		mj_step(m, d);
		
		double cart_pos = d->sensordata[m->sensor_adr[cart_pos_id]];
		double cart_vel = d->sensordata[m->sensor_adr[cart_vel_id]];
		double pole_pos = d->sensordata[m->sensor_adr[pole_pos_id]];
		double pole_vel = d->sensordata[m->sensor_adr[pole_vel_id]];
		std::cout << "cart_pos=" << cart_pos << " pole_pos=" << pole_pos << std::endl;
	}
	
	mj_deleteData(d);
	mj_deleteModel(m);
	return 0;
}

