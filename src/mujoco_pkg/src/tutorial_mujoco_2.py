# 
import time

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('../models/cart_pole.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time() #time.time 現實時間
  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync() # 滑鼠、拉桿輸入

    # Rudimentary time keeping, will drift relative to wall clock.
    # m.opt.timestep 模型的 timestep
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step) #sleep(timestep - 程式運行時間)
      