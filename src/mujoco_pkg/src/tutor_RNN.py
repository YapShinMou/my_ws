# https://github.com/udacity/deep-learning-v2-pytorch/blob/master/recurrent-neural-networks/time-series/Simple_RNN.ipynb
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

class RNN(nn.Module):
   def __init__(self, input_dim, hidden_dim, output_dim):
      super(RNN, self).__init__()
      self.hidden_dim = hidden_dim
      self.rnn = nn.RNN(input_dim, hidden_dim, 1, batch_first=True)
      self.fc = nn.Linear(hidden_dim, output_dim)

   def forward(self, x, hidden):
      # x (batch_size, seq_length, input_dim)
      r_out, hidden = self.rnn.forward(x, hidden)
      # r_out (batch_size, seq_length, hidden_dim)

      last_out = r_out[:, -1, :]
      # last_out (batch_size, hidden_dim)
      
      output = self.fc(last_out)
      # output (batch_size, output_dim)
      
      return output, hidden


class ReplayBuffer:
   def __init__(self, memory_size, seq_length):
      self.buffer = []
      self.memory_size = memory_size
      self.seq_length = seq_length

   def push(self, input, target):
      if len(self.buffer) >= self.memory_size:
         self.buffer.pop(0)
      self.buffer.append((input, target))

   def sample(self, batch_size):
      if len(self.buffer) < self.seq_length + batch_size:
         return

      indices = random.sample(range(self.seq_length - 1, len(self.buffer)), batch_size)
      
      batch_inputs = []
      batch_targets = []

      for idx in indices:
         sequence_data = self.buffer[idx - self.seq_length + 1 : idx + 1]
         
         input = np.array([s[0] for s in sequence_data])
         target = sequence_data[-1][1]

         batch_inputs.append(input)
         batch_targets.append(target)

      # batch_inputs (batch_size, seq_length, input_dim)
      # batch_targets (batch_size, target_dim)
      return np.array(batch_inputs), np.array(batch_targets)

   def size(self):
      return len(self.buffer)


class deep_learning:
   def __init__(self):
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.rnn = RNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(self.device)
      self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=LR)
      self.loss_fn = nn.MSELoss()

   def net_forward(self, input):
      with torch.no_grad():
         input_tensor = torch.tensor(input, dtype=torch.float32).to(self.device)
         prediction, _ = self.rnn.forward(input_tensor, None)
      return prediction.squeeze().cpu().numpy()

   def train(self, experience_repository):
      if experience_repository.size() < BATCH_SIZE + SEQUENCE_LENGTH:
         return 0

      inputs, targets = experience_repository.sample(BATCH_SIZE)

      x_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
      y_tensor = torch.tensor(targets, dtype=torch.float32).to(self.device)

      prediction, _ = self.rnn.forward(x_tensor, None)

      loss = self.loss_fn(prediction, y_tensor)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

   def save_model(self):
      torch.jit.save(self.rnn, 'RNN.pt')


def main():
   DL = deep_learning()
   memory = ReplayBuffer(MEMORY_SIZE, SEQUENCE_LENGTH)

   for t in range(1000):
      pos = t * t * 0.5 * 0.000001
      vel = t * 0.001
      acc = 1

      target_1 = (t+1) * (t+1) * 0.5 * 0.0001
      target_2 = target_1 + 10

      state = np.array([pos, vel, acc])
      target = np.array([target_1, target_2])

      memory.push(state, target)

   for i in range(1000):
      DL.train(memory)

   inputs, targets = memory.sample(1)
   prediction = DL.net_forward(inputs)
   print(f"inputs: {inputs}")
   print(f"targets: {targets}")
   print(f"prediction: {prediction}")
   #DL.save_model()

main()
