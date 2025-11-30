#
import numpy as np
import collections
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('../models/cart_pole.xml')
d = mujoco.MjData(m)

# --- 超參數 ---
STATE_DIM = 4
ACTION_DIM = 1
MEMORY_SIZE = 500
BATCH_SIZE = 128
EPISODES = 300

GAMMA = 0.99
TAU = 0.005
LR_Q = 1e-4
LR_PI = 1e-6
LR_ALPHA = 3e-4
ALPHA = 0.001

# --- MLP ---
class SimpleNeuralNetwork(nn.Module):
   def __init__(self, input_dim, hidden_dim, output_dim):
      super().__init__()
      self.layer1 = nn.Linear(input_dim, hidden_dim)
      self.layer2 = nn.Linear(hidden_dim, hidden_dim)
      self.layer3 = nn.Linear(hidden_dim, output_dim)

   def forward(self, x):
      x = F.relu(self.layer1(x))
      x = F.relu(self.layer2(x))
      output = self.layer3(x)
      return output

# --- ReplayBuffer ---
class ReplayBuffer:
   def __init__(self):
      self.buffer_ = collections.deque()

   def push(self, state, action, reward, next_state, not_terminal):
      # print(f"push state: \n{state}")
      # print(f"push action: \n{action}")
      # print(f"push next_state: \n{next_state}")
      experience = (state, action, reward, next_state, not_terminal)
      if len(self.buffer_) >= MEMORY_SIZE:
         self.buffer_.pop()
      self.buffer_.appendleft(experience)

   def sample(self):
      batch = random.sample(self.buffer_, BATCH_SIZE)
      # print(f"sample batch: \n{batch}")
      return batch

   def size(self):
      return len(self.buffer_)
      
# --- SAC ---
class deep_learning:
   log_2pi = math.log(2 * math.pi)

   def __init__(self): # 缺alpha_net
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print(f"Using {self.device} device")

      self.q1_net = SimpleNeuralNetwork(STATE_DIM + ACTION_DIM, 128, 1).to(self.device)
      self.q2_net = SimpleNeuralNetwork(STATE_DIM + ACTION_DIM, 128, 1).to(self.device)
      self.q1_target = SimpleNeuralNetwork(STATE_DIM + ACTION_DIM, 128, 1).to(self.device)
      self.q2_target = SimpleNeuralNetwork(STATE_DIM + ACTION_DIM, 128, 1).to(self.device)
      self.policy_net = SimpleNeuralNetwork(STATE_DIM, 128, 2 * ACTION_DIM).to(self.device)
      self.log_alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=self.device))

      self.opt_q1 = torch.optim.Adam(self.q1_net.parameters(), lr = LR_Q)
      self.opt_q2 = torch.optim.Adam(self.q2_net.parameters(), lr = LR_Q)
      self.opt_pi = torch.optim.Adam(self.policy_net.parameters(), lr = LR_PI)
      self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)
      self.loss_fn = nn.MSELoss()

      with torch.no_grad():
         self.q1_target.load_state_dict(self.q1_net.state_dict())
         self.q2_target.load_state_dict(self.q2_net.state_dict())

   def select_action(self, state):
      with torch.no_grad():
         state_tensor = torch.from_numpy(state).float().to(self.device)
         action_tensor, log_pi = self.sample_action(state_tensor)
         action_tensor = action_tensor.view(-1)
         action = action_tensor.cpu()
         action = action.numpy()
         #print(f"action: {action}")
         return action

   def train(self, experience_repository):
      if experience_repository.size() < BATCH_SIZE:
         return 0

      batch = experience_repository.sample()
      state, action, reward, next_state, not_terminal = zip(*batch)
      # print(f"state: \n{state}")
      state_batch = torch.from_numpy(np.array(state)).float().to(self.device)
      action_batch = torch.from_numpy(np.array(action)).float().to(self.device)
      # print(f"state_batch: \n{state_batch}")
      # print(f"action_batch: \n{action_batch}")
      reward_batch_ = torch.from_numpy(np.array(reward)).float().to(self.device)
      reward_batch = reward_batch_.reshape((BATCH_SIZE, 1))
      next_state_batch = torch.from_numpy(np.array(next_state)).float().to(self.device)
      not_t_batch_ = torch.from_numpy(np.array(not_terminal)).float().to(self.device)
      not_t_batch = not_t_batch_.reshape((BATCH_SIZE, 1))

      alpha = self.log_alpha.exp()

      # --- Q ---
      next_action, next_log_pi = self.sample_action(next_state_batch)
      # print(torch.cat((state_batch, action_batch), 1))
      next_q1 = self.q1_target.forward(torch.cat((next_state_batch, next_action), 1))
      next_q2 = self.q2_target.forward(torch.cat((next_state_batch, next_action), 1))
      q_target_min = torch.min(next_q1, next_q2)
      q_target = reward_batch + GAMMA * not_t_batch * (q_target_min - alpha * next_log_pi)

      q1 = self.q1_net.forward(torch.cat((state_batch, action_batch), 1))
      q2 = self.q2_net.forward(torch.cat((state_batch, action_batch), 1))
      self.opt_q1.zero_grad()
      self.opt_q2.zero_grad()
      #print(f"q1: \n{q1}")
      #print(f"q_target: \n{q_target}")
      loss_q1 = self.loss_fn(q1, q_target.detach())
      loss_q2 = self.loss_fn(q2, q_target.detach())
      loss_q1.backward()
      loss_q2.backward()
      self.opt_q1.step()
      self.opt_q2.step()
      self.q1_net.eval()
      self.q2_net.eval()

      # --- policy ---
      a_pi, log_pi = self.sample_action(state_batch)
      q_input_pi = torch.cat((state_batch, a_pi), 1)
      q_min_pi = torch.min(self.q1_net.forward(q_input_pi), self.q2_net.forward(q_input_pi))
      loss_pi = (alpha.detach() * log_pi - q_min_pi).mean()
      #print(f"loss: \n{loss_pi}")

      self.opt_pi.zero_grad()
      loss_pi.backward()
      self.opt_pi.step()

      # --- alpha ---
      alpha_loss = (alpha * (-log_pi.detach() + ACTION_DIM)).mean()
      self.opt_alpha.zero_grad()
      alpha_loss.backward()
      self.opt_alpha.step()

      # --- Q target ---
      self.update_target_net()

   def sample_action(self, state):
      mean_logsigma = self.policy_net.forward(state) #.reshape((BATCH_SIZE, 1))
      #print(f"mean_logsigma: \n{mean_logsigma}")
      check_vector = torch.from_numpy(np.zeros(2*ACTION_DIM, dtype=np.float32)).float().to(self.device)
      #print(f"mean_logsigma {mean_logsigma}")
      #print(f"check_vector {check_vector}")
      if mean_logsigma.shape == check_vector.shape:
         mean_logsigma = mean_logsigma.reshape((1, 2*ACTION_DIM))
         #print(f"mean_logsigma: \n{mean_logsigma}")

      mean = mean_logsigma[:, 0:ACTION_DIM]
      log_sigma = torch.clamp(mean_logsigma[:, ACTION_DIM:ACTION_DIM + ACTION_DIM], -20, 3)
      sigma_ = torch.exp(log_sigma)
      #print(f"mean {mean}")
      #print(f"sigma_ {sigma_}")
      z = torch.normal(mean, sigma_)

      action = torch.tanh(z)
      log_pi_z = -0.5 * (((z - mean) / sigma_).pow(2) + 2 * log_sigma + self.log_2pi)
      correction_term = torch.log(1.0 - action.pow(2) + 0.00001)
      log_pi = log_pi_z.sum(1, keepdim=True) - correction_term.sum(1, keepdim=True)
      # print(log_pi)
      return action, log_pi

   def update_target_net(self):
      with torch.no_grad():
         for target1_param, q1_param in zip(self.q1_target.parameters(), self.q1_net.parameters()):
            target1_param.data.copy_((1.0 - TAU) * target1_param.data + TAU * q1_param.data)
         for target2_param, q2_param in zip(self.q2_target.parameters(), self.q2_net.parameters()):
            target2_param.data.copy_((1.0 - TAU) * target2_param.data + TAU * q2_param.data)
      return 0

# --- get_reward() ---
def get_reward(state, next_state, not_terminal):
   if abs(next_state[2]) < 1:
      return 1
   return 0

# --- main() ---
def main():
   DL = deep_learning()
   memory = ReplayBuffer()

   with mujoco.viewer.launch_passive(m, d) as viewer:
      for episode in range(EPISODES):
         mujoco.mj_resetData(m, d)
         mujoco.mj_forward(m, d)

         total_reward = 0
         step_count = 0
         not_terminal = 1

         cart_pos = d.sensor('cart_pos').data[0]
         cart_vel = d.sensor('cart_vel').data[0]
         pole_pos = d.sensor('pole_pos').data[0]
         pole_vel = d.sensor('pole_vel').data[0]
         state = np.array([cart_pos, cart_vel, pole_pos, pole_vel], dtype=np.float32)
         d.actuator('cart_motor').ctrl = 0.0

         while not_terminal:
            step_start = time.time()

            action = DL.select_action(state)
            d.actuator('cart_motor').ctrl = action[0] * 500

            mujoco.mj_step(m, d)
            viewer.sync()
            cart_pos = d.sensor('cart_pos').data[0]
            cart_vel = d.sensor('cart_vel').data[0]
            pole_pos = d.sensor('pole_pos').data[0]
            pole_vel = d.sensor('pole_vel').data[0]
            # print(f"cart_pos: {cart_pos}")
            next_state = np.array([cart_pos, cart_vel, pole_pos, pole_vel], dtype=np.float32)

            step_count = step_count + 1
            if step_count>4000 or cart_pos>4.5 or cart_pos<-4.5 or pole_pos<-1.5 or pole_pos>1.5:
               not_terminal = 0

            reward = get_reward(state, next_state, not_terminal)
            total_reward = total_reward + reward

            memory.push(state, action, reward, next_state, not_terminal)
            DL.train(memory)

            state = next_state

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
               time.sleep(time_until_next_step * 0.2)
         print(f"Episode {episode}, Steps: {step_count}, Reward: {total_reward}")

main()