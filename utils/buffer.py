import random
from collections import deque
import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, buffer_size, num_agents):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.buffers = [deque(maxlen=buffer_size) for _ in range(num_agents)]

    def add(self, agent_index, state, action, reward, next_state):
        self.buffers[agent_index].append((state, action, reward, next_state))

    def sample(self, agent_index, batch_size):
        buffer = self.buffers[agent_index]  
        if len(buffer) < batch_size:
            return None  # Not enough samples to draw a batch
        samples = random.sample(buffer, batch_size)
        states, actions, rewards, next_states = zip(*samples)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)

    def size(self):
        return [len(buffer) for buffer in self.buffers]