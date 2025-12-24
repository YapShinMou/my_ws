# https://ithelp.ithome.com.tw/articles/10361001 
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM = 3
OUTPUT_DIM = 2
HIDDEN_DIM = 64
SEQUENCE_LENGTH = 4
MEMORY_SIZE = 100
BATCH_SIZE = 16
LR = 0.001


class LSTM(nn.Module):
   def __init__(self, input_dim, hidden_size, output_dim, num_layers=2):
      super(LSTM, self).__init__()
      self.hidden_size = hidden_size
      self.num_layers = num_layers
      
      self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
      self.fc = nn.Linear(hidden_size, 2)
      
   def forward(self, x, hidden=None):
      # x (batch_size, seq_length, input_dim)
      if hidden is None:
         # h0 (num_layers, batch_size, hidden_size)
         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
         hidden = (h0, c0)

      out, next_hidden = self.lstm(x, hidden)
      out = self.fc(out[:, -1, :])
      # out (batch_size, output_dim)
      return out, next_hidden


class ReplayBuffer:
   def __init__(self, memory_size, seq_length):
      self.input_buffer = []
      self.target_buffer = []
      self.memory_size = memory_size
      self.seq_length = seq_length

   def push(self, input, target):
      if len(self.input_buffer) >= self.memory_size:
         self.input_buffer.pop(0)
         self.target_buffer.pop(0)
      self.input_buffer.append(input)
      self.target_buffer.append(target)

   def sample(self, batch_size):
      if len(self.input_buffer) < self.seq_length + batch_size:
         return None, None

      batch_inputs = []
      batch_targets = []

      for i in range(batch_size):
         indices = random.randrange(0, len(self.input_buffer) - self.seq_length + 1)

         input = self.input_buffer[indices : indices + self.seq_length]
         target = self.target_buffer[indices : indices + self.seq_length]
         input.reverse()
         target.reverse()
         batch_inputs.append(input)
         batch_targets.append(target)

      return np.array(batch_inputs), np.array(batch_targets)

   def size(self):
      return len(self.input_buffer)


class deep_learning:
   def __init__(self, input_dim, hidden_dim, output_dim):
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.lstm = LSTM(input_dim, hidden_dim, output_dim).to(self.device)
      self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=LR)
      self.loss_fn = nn.MSELoss()
      self.input_buffer = []

   def net_forward(self, input, reset=False):
      if reset == True:
         self.input_buffer = []
      self.input_buffer.append(input)

      if len(self.input_buffer) > SEQUENCE_LENGTH:
         self.input_buffer.pop(0)
         with torch.no_grad():
            input_tensor = torch.tensor(self.input_buffer, dtype=torch.float32).to(self.device)
            input_tensor = input_tensor.unsqueeze(0)
            prediction, _ = self.lstm.forward(input_tensor)

         # prediction.squeeze() (output_dim)
         return prediction.squeeze().cpu().numpy()

   def train(self, buffer):
      if buffer.size() < BATCH_SIZE + SEQUENCE_LENGTH:
         return 0

      inputs, targets = buffer.sample(BATCH_SIZE)
      x_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
      y_tensor = torch.tensor(targets, dtype=torch.float32).to(self.device)
      y_tensor = y_tensor[:, -1, :]
      prediction, _ = self.lstm.forward(x_tensor)

      self.optimizer.zero_grad()
      loss = self.loss_fn(prediction, y_tensor)
      loss.backward()
      self.optimizer.step()


def main():
   DL = deep_learning(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
   memory = ReplayBuffer(MEMORY_SIZE, SEQUENCE_LENGTH)

   for t in range(1000):
      pos = t * t * 0.5 * 0.000001
      vel = t * 0.001
      acc = 1

      target_1 = (t+1) * (t+1) * 0.5 * 0.000001
      target_2 = target_1 + 0.5

      state = np.array([pos, vel, acc])
      target = np.array([target_1, target_2])

      memory.push(state, target)

   for epoch in range(5000):
      DL.train(memory)

   next_hidden = None
   for t in range(20):
      t = t+100
      pos = t * t * 0.5 * 0.000001
      vel = t * 0.001
      acc = 1

      target_1 = (t+1) * (t+1) * 0.5 * 0.000001
      target_2 = target_1 + 0.5

      state = np.array([pos, vel, acc])
      target = np.array([target_1, target_2])
      prediction = DL.net_forward(state)
      print(f"traget: {target}")
      print(f"prediction: {prediction}")

main()
