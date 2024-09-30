import random
import numpy as np
from collections import namedtuple
import torch

# 每个 “状态转换” 包括5元组: 状态 策略 奖励 下一状态 是否结束
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        # state, action, reward, next_state, done = map(np.stack, zip(*batch))
        # state = np.concatenate([np.expand_dims(state, 0) for state, _, _, _, _ in batch], axis=0)
        # action = np.concatenate([np.expand_dims(action, 0) for _, action, _, _, _ in batch], axis=0)
        # reward = np.concatenate([np.expand_dims(reward, 0) for _, _, reward, _, _ in batch], axis=0)
        # next_state = np.concatenate([np.expand_dims(next_state, 0) for _, _, _, next_state, _ in batch], axis=0)
        # done = np.concatenate([np.expand_dims(done, 0) for _, _, _, _, done in batch], axis=0)

        # batch = Transition(*zip(*batch))
        #
        # states = torch.cat(batch.state)
        # actions = torch.cat(batch.action)
        # rewards = torch.cat(batch.reward)
        # next_states = torch.cat(batch.next_state)
        # dones = torch.cat(batch.done)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
