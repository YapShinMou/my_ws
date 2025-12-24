# 
import time
import yaml

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('../models/cube.xml')
d = mujoco.MjData(m)

paused = False

def key_callback(keycode):
   if chr(keycode) == ' ':
      global paused
      paused = not paused

with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
   while viewer.is_running():
      step_start = time.time()
      with open('set.yaml', 'r') as f:
         y = yaml.safe_load(f)
      if y is None:
         y = {'x': 0, 'y': 0, 'z': 0}
      xf = y['x']
      yf = y['y']
      zf = y['z']

      # d.actuator('cart_motor').ctrl = forc

      if not paused:
         mujoco.mj_step(m, d)
         viewer.sync()

      #cart_pos = d.sensor('cart_pos').data[0]

      time_until_next_step = m.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
         time.sleep(time_until_next_step)
      