import numpy as np
from collections import deque


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, epsilon=1):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = beta

    def add(self, actor, state, action, next_state, extrinsic_reward, intrinsic_reward, priority):
        # Check if buffer is full before adding
        if len(self.buffer) < self.capacity:
            self.buffer.append((actor, state, action, next_state, extrinsic_reward, intrinsic_reward))
            self.priorities.append(priority + self.epsilon)
        else:
            # If full, remove the oldest element (FIFO)
            self.buffer.popleft()
            self.priorities.popleft()
            # Add the new element
            self.buffer.append((actor, state, action, next_state, extrinsic_reward, intrinsic_reward))
            self.priorities.append(priority + self.epsilon)

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], [], [], [], []
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        return experiences, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + self.epsilon


def sample_n(mu_t, sigma, n):
    return list(np.clip([np.random.normal(mu_t, sigma) for _ in range(n)], 0, 1))


def get_loss(Q, s, s_next, a, r, discount_f):
    best_next_action = np.argmax(Q[s_next])
    td_target = r + discount_f * Q[s_next][best_next_action]
    return abs(td_target - Q[s][a])
