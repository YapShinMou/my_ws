# simple check model
import mujoco
import mujoco.viewer as viewer

# Make model and data
model = mujoco.MjModel.from_xml_path('../models/g1_description/scene.xml')
#model = mujoco.MjModel.from_xml_path('unitree_mujoco/unitree_robots/g1/scene.xml')
data = mujoco.MjData(model)

viewer.launch(model, data)
