# 2026/1/31 範例
import numpy as np
import collections
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
   def __init__(self, input_dim, hidden_dim, output_dim):
      super().__init__()
      self.layer1 = nn.Linear(input_dim, hidden_dim)
      self.layer2 = nn.Linear(hidden_dim, hidden_dim)
      self.layer3 = nn.Linear(hidden_dim, output_dim)

   def forward(self, x):
      x = self.layer1(x)
      x = F.relu(x)
      x = self.layer2(x)
      x = F.relu(x)
      return self.layer3(x)


class ReplayBuffer:
   def __init__(self, input_dim, target_dim, memory_size, batch_size):
      self.input_dim = input_dim
      self.target_dim = target_dim
      self.input_buffer = np.zeros((memory_size, input_dim), dtype=np.float32)
      self.target_buffer = np.zeros((memory_size, target_dim), dtype=np.float32)
      self.memory_size = memory_size
      self.batch_size = batch_size
      self.pointer = 0
      self.size = 0

   def push(self, input, target): # input np([  ])
      self.input_buffer[self.pointer] = input
      self.target_buffer[self.pointer] = target
      self.pointer = (self.pointer + 1) % self.memory_size
      if self.size < self.memory_size:
         self.size = self.size + 1

   def sample(self):
      if self.size < self.batch_size:
         return None, None
      indices = np.random.randint(0, self.size, size=self.batch_size)
      input_batch = self.input_buffer[indices]
      target_batch = self.target_buffer[indices]
      return input_batch, target_batch

   def save_buffer(self, file_name):
      input_buffer_ = np.resize(self.input_buffer, (self.size, self.input_dim))
      target_buffer_ = np.resize(self.target_buffer, (self.size, self.target_dim))
      np.savez(file_name, a=input_buffer_, b=target_buffer_)
      print(f"save buffer: {file_name}")

   def load_buffer(self, file_name):
      buffer = np.load(file_name)
      self.input_buffer = np.resize(self.input_buffer, (self.size, self.input_dim))
      self.target_buffer = np.resize(self.target_buffer, (self.size, self.target_dim))
      self.input_buffer = np.concatenate((buffer['a'], self.input_buffer))
      self.target_buffer = np.concatenate((buffer['b'], self.target_buffer))
      if self.input_buffer.shape[0] < self.memory_size:
         self.pointer = self.input_buffer.shape[0]
         self.size = self.pointer
         self.input_buffer = np.concatenate((self.input_buffer, np.zeros((self.memory_size - self.input_buffer.shape[0], self.input_dim), dtype=np.float32)))
         self.target_buffer = np.concatenate((self.target_buffer, np.zeros((self.memory_size - self.target_buffer.shape[0], self.target_dim), dtype=np.float32)))
         self.size = self.input_buffer.shape[0]
         print(f"load buffer")
      else:
         self.input_buffer = self.input_buffer[self.input_buffer.shape[0] - self.memory_size:self.input_buffer.shape[0], :]
         self.target_buffer = self.target_buffer[self.target_buffer.shape[0] - self.memory_size:self.target_buffer.shape[0], :]
         self.pointer = 0
         self.size = self.memory_size
         print(f"load buffer, buffer overflow")


class deep_learning:
   def __init__(self, input_dim, hidden_dim, output_dim):
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print(f"Using {self.device} device")
      self.net_1 = NeuralNetwork(input_dim, hidden_dim, output_dim).to(self.device)
      self.optimizer_1 = torch.optim.Adam(self.net_1.parameters(), lr=0.001)
      self.loss_fn = nn.MSELoss()

   def predict(self, input):
      with torch.no_grad():
         input_tensor = torch.tensor(input, dtype=torch.float32).to(self.device)
         output = self.net_1.forward(input_tensor)
      return output.squeeze().cpu().numpy()

   def train(self, buffer):
      input_batch, target_batch = buffer.sample()
      if input_batch is None:
         print(f"buffer 不夠大，跳過訓練")
         return
      input_torch = torch.tensor(input_batch, dtype=torch.float32).to(self.device)
      target_torch = torch.tensor(target_batch, dtype=torch.float32).to(self.device)

      output_torch = self.net_1.forward(input_torch)

      self.optimizer_1.zero_grad()
      self.loss = self.loss_fn(output_torch, target_torch)
      self.loss.backward()
      self.optimizer_1.step()

   def save_model(self, file_name):
      torch.save(self.net_1, file_name)
      print(f"save model: {file_name}")

   def load_model(self, file_name):
      self.net_1 = torch.load(file_name, weights_only=False)
      self.net_1.to(self.device)
      self.optimizer_1 = torch.optim.Adam(self.net_1.parameters(), lr=0.001)
      self.net_1.train()
      print(f"load model: {file_name}")


def main():
   INPUT_DIM = 4
   OUTPUT_DIM = 2
   MEMORY_SIZE = 50
   BATCH_SIZE = 32

   DL = deep_learning(INPUT_DIM, 32, OUTPUT_DIM)
   # DL.load_model("net_1.pt")
   memory = ReplayBuffer(INPUT_DIM, OUTPUT_DIM, MEMORY_SIZE, BATCH_SIZE)
   # memory.load_buffer('buffer.npz')

   a_input = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
   a_target = np.array([3.0, 7.0], dtype=np.float32)
   b_input = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
   b_target = np.array([4.0, 8.0], dtype=np.float32)
   c_input = np.array([3, 5, 7, 9], dtype=np.float32)
   c_target = np.array([8, 16], dtype=np.float32)
   print(f"target: {a_target} {b_target} {c_target}")

   output_a = DL.predict(a_input)
   output_b = DL.predict(b_input)
   output_c = DL.predict(c_input)
   print(f"untrain output: {output_a} {output_b} {output_c}")

   for i in range(40):
      intput_1 = np.random.random()*20 - 10
      intput_2 = np.random.random()*20 - 10
      intput_3 = np.random.random()*20 - 10
      intput_4 = np.random.random()*20 - 10
      input = np.array([intput_1, intput_2, intput_3, intput_4], dtype=np.float32)
      target = np.array([intput_1+intput_2+np.random.normal(), intput_3+intput_4+np.random.normal()], dtype=np.float32)
      memory.push(input, target)

   for i in range(5000):
      DL.train(memory)

   output_a = DL.predict(a_input)
   output_b = DL.predict(b_input)
   output_c = DL.predict(c_input)
   print(f"trained output: {output_a} {output_b} {output_c}")

   # DL.save_model("net_1.pt")
   # memory.save_buffer('buffer.npz')


start_time = time.time()
main()
print(f"run time: {time.time() - start_time}")