# simple without GUI
import numpy as np
import mujoco

m = mujoco.MjModel.from_xml_path('../models/cart_pole.xml')
d = mujoco.MjData(m)

for i in range(2):
   mujoco.mj_resetData(m, d)

   while d.time < 1:
      d.actuator('cart_motor').ctrl = 50
      mujoco.mj_step(m, d)

      CART_POS = d.sensor('cart_pos').data
      CART_VEL = d.sensor('cart_vel').data
      POLE_POS = d.sensor('pole_pos').data
      POLE_VEL = d.sensor('pole_vel').data

      print(CART_POS, CART_VEL, POLE_POS, POLE_VEL)
