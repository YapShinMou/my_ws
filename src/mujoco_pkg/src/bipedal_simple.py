
import time
import yaml
import math

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('../models/bipedal_scene.xml')
d = mujoco.MjData(m)

paused = False

def key_callback(keycode):
  if chr(keycode) == ' ':
    global paused
    paused = not paused

with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
  start = time.time() #time.time 現實時間
  sin_time = 0
  stop_time = 0

  while viewer.is_running():
    step_start = time.time()
    with open('set.yaml', 'r') as f:
      y = yaml.safe_load(f)
    if y is None:
      y = {'left_hip_pitch': 0, 
            'left_hip_roll': 0,
            'left_hip_yaw': 0,
            'left_knee': 0,
            'left_ankle_pitch': 0,
            'left_ankle_roll': 0,
            'right_hip_pitch': 0,
            'right_hip_roll': 0,
            'right_hip_yaw': 0,
            'right_knee': 0,
            'right_ankle_pitch': 0,
            'right_ankle_roll': 0}

    left_hip_pitch = y['left_hip_pitch']
    left_hip_roll = y['left_hip_roll']
    left_hip_yaw = y['left_hip_yaw']
    left_knee = y['left_knee']
    left_ankle_pitch = y['left_ankle_pitch']
    left_ankle_roll = y['left_ankle_roll']
    
    right_hip_pitch = y['right_hip_pitch']
    right_hip_roll = y['right_hip_roll']
    right_hip_yaw = y['right_hip_yaw']
    right_knee = y['right_knee']
    right_ankle_pitch = y['right_ankle_pitch']
    right_ankle_roll = y['right_ankle_roll']

    sin_amp = math.sin((math.pi/1000) * sin_time)

    if sin_time >= 200 and sin_time <= 400:
      left_amp = math.cos((math.pi/200) * (sin_time - 200))
    elif sin_time > 400 and sin_time < 600:
      left_amp = -1
    elif sin_time >= 600 and sin_time <= 800:
      left_amp = math.cos((math.pi/200) * sin_time)
    else:
      left_amp = 1

    sin_time = sin_time + 1
    if sin_time >= 2000:
      sin_time = 0

    if stop_time < 1000:
      stop_time = stop_time + 1
      sin_amp = 0
      left_amp = 1

    #d.actuator('left_hip_pitch').ctrl = left_amp * 0.3 - 0.3
    #d.actuator('left_hip_roll').ctrl = sin_amp * 0.08
    #d.actuator('left_hip_yaw').ctrl = left_hip_yaw
    #d.actuator('left_knee').ctrl = 0.5 - (left_amp * 0.5)
    #d.actuator('left_ankle_pitch').ctrl = left_amp * 0.2 - 0.2
    #d.actuator('left_ankle_roll').ctrl = -(sin_amp * 0.08)

    #d.actuator('right_hip_pitch').ctrl = right_hip_pitch
    #d.actuator('right_hip_roll').ctrl = sin_amp * 0.08
    #d.actuator('right_hip_yaw').ctrl = right_hip_yaw
    #d.actuator('right_knee').ctrl = right_knee
    #d.actuator('right_ankle_pitch').ctrl = right_ankle_pitch
    #d.actuator('right_ankle_roll').ctrl = -(sin_amp * 0.08)

    if not paused:
      mujoco.mj_step(m, d)
      viewer.sync() # 滑鼠、拉桿輸入

    # 腿部關節 (Leg Joints)
    #left_hip_pitch_pos = d.sensor('left_hip_pitch_pos').data[0]
    #left_hip_roll_pos = d.sensor('left_hip_roll_pos').data[0]
    #left_hip_yaw_pos = d.sensor('left_hip_yaw_pos').data[0]
    #left_knee_pos = d.sensor('left_knee_pos').data[0]
    #left_ankle_pitch_pos = d.sensor('left_ankle_pitch_pos').data[0]
    #left_ankle_roll_pos = d.sensor('left_ankle_roll_pos').data[0]

    #right_hip_pitch_pos = d.sensor('right_hip_pitch_pos').data[0]
    #right_hip_roll_pos = d.sensor('right_hip_roll_pos').data[0]
    #right_hip_yaw_pos = d.sensor('right_hip_yaw_pos').data[0]
    #right_knee_pos = d.sensor('right_knee_pos').data[0]
    #right_ankle_pitch_pos = d.sensor('right_ankle_pitch_pos').data[0]
    #right_ankle_roll_pos = d.sensor('right_ankle_roll_pos').data[0]

    #imu_quat_0 = d.sensor('imu_quat')
    #imu_quat_1 = d.sensor('imu_quat').data[1]
    #imu_quat_2 = d.sensor('imu_quat').data[2]
    #imu_quat_3 = d.sensor('imu_quat').data[3]

    #imu_gyro = d.sensor('imu_gyro')
    #imu_acc = d.sensor('imu_acc')
    #frame_pos = d.sensor('frame_pos')
    #frame_vel = d.sensor('frame_vel')

    # print(frame_vel)

    # m.opt.timestep 模型的 timestep
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step) #sleep(模型timestep - 程式運行時間)
      