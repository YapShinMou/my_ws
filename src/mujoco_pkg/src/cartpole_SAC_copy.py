
import numpy as np
import collections
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import time
import mujoco
# import mujoco.viewer

m = mujoco.MjModel.from_xml_path('../models/cart_pole.xml')
d = mujoco.MjData(m)

# --- 超參數 ---
STATE_DIM = 4
ACTION_DIM = 1
MEMORY_SIZE = 10000
BATCH_SIZE = 256
EPISODES = 1000

GAMMA = 0.99
TAU = 0.005
LR_CRITIC = 3e-5
LR_ACTOR = 3e-7
LR_ALPHA = 3e-6
TARGET_ENTROPY = -1

class PolicyNetContinuous(torch.nn.Module):
   def __init__(self, state_dim, hidden_dim, action_dim):
      super().__init__()
      self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
      self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
      self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

   def forward(self, x):
      x = F.relu(self.fc1(x))
      mu = self.fc_mu(x)
      std = F.softplus(self.fc_std(x))
      dist = Normal(mu, std)
      normal_sample = dist.rsample()  # rsample()是重参数化采样
      log_prob = dist.log_prob(normal_sample)
      action = torch.tanh(normal_sample)
      # 计算tanh_normal分布的对数概率密度
      log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
      return action, log_prob

class QValueNetContinuous(torch.nn.Module):
   def __init__(self, state_dim, hidden_dim, action_dim):
      super().__init__()
      self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
      self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
      self.fc_out = torch.nn.Linear(hidden_dim, 1)

   def forward(self, x, a):
      cat = torch.cat([x, a], dim=1)
      x = F.relu(self.fc1(cat))
      x = F.relu(self.fc2(x))
      return self.fc_out(x)

# --- ReplayBuffer ---
class ReplayBuffer:
   def __init__(self):
      self.buffer_ = collections.deque()

   def push(self, state, action, reward, next_state, done):
      # print(f"push state: \n{state}")
      # print(f"push action: \n{action}")
      # print(f"push next_state: \n{next_state}")
      experience = (state, action, reward, next_state, done)
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
   def __init__(self):
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print(f"Using {self.device} device")

      self.actor = PolicyNetContinuous(STATE_DIM, 256, ACTION_DIM).to(self.device)  # 策略网络
      self.critic_1 = QValueNetContinuous(STATE_DIM, 256, ACTION_DIM).to(self.device)  # 第一个Q网络
      self.critic_2 = QValueNetContinuous(STATE_DIM, 256, ACTION_DIM).to(self.device)  # 第二个Q网络
      self.target_critic_1 = QValueNetContinuous(STATE_DIM, 256, ACTION_DIM).to(self.device)  # 第一个目标Q网络
      self.target_critic_2 = QValueNetContinuous(STATE_DIM, 256, ACTION_DIM).to(self.device)  # 第二个目标Q网络

      with torch.no_grad():
         self.target_critic_1.load_state_dict(self.critic_1.state_dict())
         self.target_critic_2.load_state_dict(self.critic_2.state_dict())

      self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
      self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=LR_CRITIC)
      self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=LR_CRITIC)

      self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
      self.log_alpha.requires_grad = True  # 可以对alpha求梯度
      self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)

      self.loss_fn = nn.MSELoss()

   def select_action(self, state):
      with torch.no_grad():
         state_tensor = torch.from_numpy(state).float().to(self.device)
         action_tensor, log_prob = self.actor.forward(state_tensor)
         # action_tensor = action_tensor.view(-1)
         action = action_tensor.cpu()
         action = action.numpy()
         return action

   def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
      next_actions, log_prob = self.actor(next_states)
      entropy = -log_prob
      q1_value = self.target_critic_1(next_states, next_actions)
      q2_value = self.target_critic_2(next_states, next_actions)
      next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
      td_target = rewards + GAMMA * next_value * (1 - dones)
      return td_target

   def soft_update(self, net, target_net):
      with torch.no_grad():
         for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - TAU) + param.data * TAU)

   def update(self, experience_repository):
      if experience_repository.size() < BATCH_SIZE:
         return 0

      batch = experience_repository.sample()
      state, action, reward, next_state, done = zip(*batch)

      state_batch = torch.from_numpy(np.array(state)).float().to(self.device)
      action_batch = torch.from_numpy(np.array(action)).float().to(self.device)
      reward_batch_ = torch.from_numpy(np.array(reward)).float().to(self.device)
      reward_batch = reward_batch_.reshape((BATCH_SIZE, 1))
      next_state_batch = torch.from_numpy(np.array(next_state)).float().to(self.device)
      done_batch_ = torch.from_numpy(np.array(done)).float().to(self.device)
      done_batch = done_batch_.reshape((BATCH_SIZE, 1))

      td_target = self.calc_target(reward_batch, next_state_batch, done_batch)
      # 算均方根損失和原版不一樣
      critic_1_loss = self.loss_fn(self.critic_1(state_batch, action_batch), td_target.detach())
      critic_2_loss = self.loss_fn(self.critic_2(state_batch, action_batch), td_target.detach())
      self.critic_1_optimizer.zero_grad()
      critic_1_loss.backward()
      self.critic_1_optimizer.step()
      self.critic_2_optimizer.zero_grad()
      critic_2_loss.backward()
      self.critic_2_optimizer.step()

      new_actions, log_prob = self.actor(state_batch)
      entropy = -log_prob
      q1_value = self.critic_1(state_batch, new_actions)
      q2_value = self.critic_2(state_batch, new_actions)
      actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_optimizer.step()

      alpha_loss = torch.mean((entropy - TARGET_ENTROPY).detach() * self.log_alpha.exp())
      self.log_alpha_optimizer.zero_grad()
      alpha_loss.backward()
      self.log_alpha_optimizer.step()

      self.soft_update(self.critic_1, self.target_critic_1)
      self.soft_update(self.critic_2, self.target_critic_2)

# --- get_reward() ---
def get_reward(state, next_state, done):
   cart_reward = (4.51 - abs(next_state[0])) / 4.5
   pole_reward = 1 - abs(next_state[2])
   return cart_reward + pole_reward

# --- main() ---
def main():
   DL = deep_learning()
   memory = ReplayBuffer()

   """
   with mujoco.viewer.launch_passive(m, d) as viewer:
      for episode in range(EPISODES):
         mujoco.mj_resetData(m, d)
         mujoco.mj_forward(m, d)

         total_reward = 0
         step_count = 0
         done = 0

         cart_pos = d.sensor('cart_pos').data[0]
         cart_vel = d.sensor('cart_vel').data[0]
         pole_pos = d.sensor('pole_pos').data[0]
         pole_vel = d.sensor('pole_vel').data[0]
         state = np.array([cart_pos, cart_vel, pole_pos, pole_vel], dtype=np.float32)
         d.actuator('cart_motor').ctrl = 0.0

         while done == 0:
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
               done = 1

            reward = get_reward(state, next_state, done)
            total_reward = total_reward + reward

            memory.push(state, action, reward, next_state, done)
            DL.update(memory)

            state = next_state

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
               time.sleep(time_until_next_step * 0.2)
         print(f"Episode {episode}, Steps: {step_count}, Reward: {total_reward}")
   """

   for episode in range(EPISODES):
      mujoco.mj_resetData(m, d)
      mujoco.mj_forward(m, d)

      total_reward = 0
      step_count = 0
      done = 0

      cart_pos = d.sensor('cart_pos').data[0]
      cart_vel = d.sensor('cart_vel').data[0]
      pole_pos = d.sensor('pole_pos').data[0]
      pole_vel = d.sensor('pole_vel').data[0]
      state = np.array([cart_pos, cart_vel, pole_pos, pole_vel], dtype=np.float32)
      d.actuator('cart_motor').ctrl = 0.0

      while done == 0:
         step_start = time.time()

         action = DL.select_action(state)
         d.actuator('cart_motor').ctrl = action[0] * 500

         mujoco.mj_step(m, d)
         cart_pos = d.sensor('cart_pos').data[0]
         cart_vel = d.sensor('cart_vel').data[0]
         pole_pos = d.sensor('pole_pos').data[0]
         pole_vel = d.sensor('pole_vel').data[0]
         next_state = np.array([cart_pos, cart_vel, pole_pos, pole_vel], dtype=np.float32)

         step_count = step_count + 1
         if step_count>4000 or cart_pos>4.5 or cart_pos<-4.5 or pole_pos<-1 or pole_pos>1:
            done = 1

         reward = get_reward(state, next_state, done)
         total_reward = total_reward + reward

         memory.push(state, action, reward, next_state, done)
         DL.update(memory)

         state = next_state

      print(f"Episode {episode}, Steps: {step_count}, Reward: {total_reward}")

main()