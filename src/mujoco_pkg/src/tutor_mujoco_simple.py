# 
import time
import yaml

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('../models/cart_pole.xml')
d = mujoco.MjData(m)

paused = False

def key_callback(keycode):
   if chr(keycode) == ' ':
      global paused
      paused = not paused

with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
   # Close the viewer automatically after 30 wall-seconds.
   start = time.time() #time.time 現實時間
   while viewer.is_running(): # and time.time() - start < 30:
      step_start = time.time()
      with open('set.yaml', 'r') as f:
         y = yaml.safe_load(f)
      if y is None:
         y = {'pos': 0, 'forc': 0}
      pos = y['pos']
      forc = y['forc']

      d.actuator('cart_motor').ctrl = forc
      d.actuator('cart_position').ctrl = pos

      if not paused:
         # mj_step can be replaced with code that also evaluates
         # a policy and applies a control signal before stepping the physics.
         mujoco.mj_step(m, d)

         # Pick up changes to the physics state, apply perturbations, update options from GUI.
         viewer.sync() # 滑鼠、拉桿輸入

      #cart_pos = d.sensor('cart_pos').data[0]
      #cart_vel = d.sensor('cart_vel').data[0]
      #pole_pos = d.sensor('pole_pos').data[0]
      #pole_vel = d.sensor('pole_vel').data[0]

      # Rudimentary time keeping, will drift relative to wall clock.
      # m.opt.timestep 模型的 timestep
      time_until_next_step = m.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
         time.sleep(time_until_next_step) #sleep(timestep - 程式運行時間)
      