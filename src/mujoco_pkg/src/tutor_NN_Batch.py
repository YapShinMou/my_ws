import numpy as np
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM = 4
OUTPUT_DIM = 2
MEMORY_SIZE = 20
BATCH_SIZE = 6
LR = 0.001

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

class ReplayBuffer:
   def __init__(self):
      self.buffer_ = collections.deque()

   def push(self, input, target):
      # print(f"push input: \n{input}")
      experience = (input, target)
      if len(self.buffer_) >= MEMORY_SIZE:
         self.buffer_.pop()
      self.buffer_.appendleft(experience)

   def sample(self):
      batch = random.sample(self.buffer_, BATCH_SIZE)
      # print(f"sample batch: {batch}")
      return batch

   def size(self):
      return len(self.buffer_)

class deep_learning:
   def __init__(self):
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print(f"Using {self.device} device")
      self.net_1 = SimpleNeuralNetwork(INPUT_DIM, 64, OUTPUT_DIM).to(self.device)
      self.optimizer_1 = torch.optim.Adam(self.net_1.parameters())
      self.loss_fn = nn.MSELoss()

   def net_forward(self, input):
      with torch.no_grad():
         input_tensor = torch.from_numpy(input).float().to(self.device)
         output = self.net_1.forward(input_tensor)
         return output

   def train(self, experience_repository):
      if experience_repository.size() < BATCH_SIZE:
         return 0

      batch = experience_repository.sample()
      inputs, targets = zip(*batch)
      # print(f"inputs: {inputs}")
      # print(f"np.array: {np.array(inputs)}")
      input_batch = torch.from_numpy(np.array(inputs)).float().to(self.device)
      print(f"input_batch: \n{input_batch}")
      target_batch = torch.from_numpy(np.array(targets)).float().to(self.device)
      print(f"target_batch: \n{target_batch}")
      output_batch = self.net_1.forward(input_batch)
      # print(output_batch)

      print(input_batch.shape[0])

      self.optimizer_1.zero_grad()
      loss = self.loss_fn(output_batch, target_batch)
      loss.backward()
      self.optimizer_1.step()
      self.net_1.eval()

def main():
   DL = deep_learning()
   memory = ReplayBuffer()

   a_input = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
   a_target = np.array([3.0, 7.0], dtype=np.float32)
   b_input = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
   b_target = np.array([4.0, 8.0], dtype=np.float32)
   c_input = np.array([3, 5, 7, 9], dtype=np.float32)
   c_target = np.array([8, 16], dtype=np.float32)

   # print(a_input)
   output = DL.net_forward(a_input)
   print(f"untrain output: {output}")

   for i in range(10):
      # print(f"a_input: \n{a_input}")
      memory.push(a_input, a_target)
      memory.push(b_input, b_target)
      memory.push(c_input, c_target)

   for i in range(1):
      DL.train(memory)

   output = DL.net_forward(a_input)
   print(f"trained output: {output}")


main()
