# simple check model
import mujoco
import mujoco.viewer as viewer

# Make model and data
model = mujoco.MjModel.from_xml_path('../models/cart_pole.xml')
data = mujoco.MjData(model)

viewer.launch(model, data)
