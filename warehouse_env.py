import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2

class WarehouseEnv(gym.Env):
    def __init__(self, grid_size=6, num_obstacles=6, num_items=2, max_steps=100):
        super(WarehouseEnv, self).__init__()
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.num_items = num_items
        self.max_steps = max_steps

        self.observation_space = spaces.Box(low=0, high=5, shape=(2,), dtype=np.int32)
        self.action_space = spaces.Discrete(4)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = [0, 0]
        self.items = []
        self.obstacles = []
        self.steps = 0
        self.carrying = False
        self.delivery_zone = [self.grid_size - 1, self.grid_size - 1]
        self.just_delivered = False

        # Generate obstacles
        while len(self.obstacles) < self.num_obstacles:
            pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            if pos != self.agent_pos and pos not in self.obstacles:
                self.obstacles.append(pos)

        # Generate items
        while len(self.items) < self.num_items:
            pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            if pos != self.agent_pos and pos not in self.obstacles and pos not in self.items:
                self.items.append(pos)

        return np.array(self.agent_pos, dtype=np.int32), {}

    def step(self, action):
        self.just_delivered = False
        self.steps += 1

        if isinstance(action, np.ndarray):
            action = int(action.item())

        move = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }

        delta = move[action]
        new_pos = [self.agent_pos[0] + delta[0], self.agent_pos[1] + delta[1]]

        # Keep within bounds
        if (0 <= new_pos[0] < self.grid_size and
            0 <= new_pos[1] < self.grid_size and
            new_pos not in self.obstacles):
            self.agent_pos = new_pos

        reward = -1
        done = False

        if self.agent_pos in self.items and not self.carrying:
            self.items.remove(self.agent_pos)
            self.carrying = True
            reward = 10

        if self.agent_pos == self.delivery_zone and self.carrying:
            self.carrying = False
            reward = 100
            self.just_delivered = True

        if self.steps >= self.max_steps:
            done = True

        return np.array(self.agent_pos, dtype=np.int32), reward, done, False, {}

    def render(self, mode='human'):
        grid = np.ones((self.grid_size, self.grid_size, 3), dtype=np.uint8) * 255

        for obs in self.obstacles:
            grid[obs[0], obs[1]] = [0, 0, 0]  # black

        for item in self.items:
            grid[item[0], item[1]] = [0, 255, 0]  # green

        grid[self.delivery_zone[0], self.delivery_zone[1]] = [0, 0, 255]  # blue

        grid[self.agent_pos[0], self.agent_pos[1]] = [255, 0, 0]  # red

        img = cv2.resize(grid, (400, 400), interpolation=cv2.INTER_NEAREST)

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            cv2.imshow("Warehouse", img)
            cv2.waitKey(100)

    def close(self):
        cv2.destroyAllWindows()
