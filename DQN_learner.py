import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque


class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim1,hidden_dim2,hidden_dim3,output_dim):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU())
        self.layer2 = torch.nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU())
        self.layer3 = torch.nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU())
        self.final = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final(x)

        return x


class DQN_learner:

    def __init__(self, poss_actions, learn_rate, batchsize, gamma, max_buffer_size, device):

        self.poss_actions = poss_actions
        self.memory = deque(maxlen=max_buffer_size)
        self.batchsize = batchsize
        self.device = device
        self.gamma = gamma
        self.Q = DQN(3, 50,50,50, poss_actions).to(device)
        self.target_net = DQN(3, 50,50,50, poss_actions).to(device)
        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.Adam(self.Q.parameters(), lr=learn_rate)

    def choose_action(self, epsilon, state):
        if np.random.rand() < epsilon:
            action = np.random.randint(0, self.poss_actions)
        else:
            self.Q.train(mode=False)
            state = state.to(self.device)
            AQ = self.Q(state)
            _, action = torch.max(AQ.data, -1)
        return action

    def save(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def Sample(self):

        Batch = random.sample(self.memory, self.batchsize)
        states = np.zeros((self.batchsize, 3))
        next_states = np.zeros((self.batchsize, 3))
        actions, rewards, dones = [], [], []

        for i in range(self.batchsize):
            states[i] = Batch[i][0]
            actions.append(Batch[i][1])
            rewards.append(Batch[i][2])
            next_states[i] = Batch[i][3]
            dones.append(Batch[i][4])
        t_states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
        t_actions = torch.as_tensor(actions, dtype=torch.int64).to(self.device)
        t_rewards = torch.as_tensor(rewards, dtype=torch.float32).to(self.device)
        t_next_states = torch.as_tensor(next_states, dtype=torch.float32).to(self.device)
        t_dones = torch.as_tensor(dones, dtype=torch.int64).to(self.device)
        return t_states, t_actions, t_rewards, t_next_states, t_dones

    def Q_update(self, states, actions, rewards, next_states, dones):
        self.Q.train(mode=True)
        self.target_net.train(mode=False)
        output = self.Q(states)
        actions = actions.reshape(output.size(0), -1)
        predict = torch.gather(output, 1, actions)
        t_output = self.target_net(next_states)
        next_target, _ = torch.max(t_output, 1, True)
        rewards = rewards.reshape(next_target.size(0), -1)
        dones = dones.reshape(next_target.size(0), -1)
        target = self.gamma * next_target * (1 - dones) + rewards
        self.optim.zero_grad()
        loss = self.loss_fn(predict, target)
        loss.backward()
        self.optim.step()

    def target_net_update(self):
        torch.save(self.Q.state_dict(), "model.pth")
        self.target_net.load_state_dict(torch.load("model.pth"))
        self.target_net.eval()


