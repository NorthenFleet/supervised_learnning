import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np


class Env:
    def __init__(self):
        self.num_drones = None
        self.num_tasks = None
        self.reset()

    def reset(self):
        self.drones = [
            {
                "position": np.random.rand(2),
                "range": np.random.rand() * 100,
                "speed": np.random.rand() * 10,
                "endurance": np.random.rand() * 100
            }
            for _ in range(self.num_drones)
        ]
        self.tasks = [
            {
                "distance": np.random.rand() * 100,
                "area_size": np.random.rand() * 100,
                "priority": random.randint(1, 10)
            }
            for _ in range(self.num_tasks)
        ]
        return self._get_state()

    def _get_state(self):
        drone_features = torch.tensor(
            [[d["position"][0], d["position"][1], d["range"],
                d["speed"], d["endurance"]] for d in self.drones],
            dtype=torch.float
        )
        task_features = torch.tensor(
            [[t["distance"], t["area_size"], t["priority"]]
                for t in self.tasks],
            dtype=torch.float
        )
        return drone_features, task_features

    def step(self, action):
        # Define the reward based on the action taken
        # This is a placeholder; in practice, this should be calculated based on task completion, drone performance, etc.
        reward = random.random()
        done = True  # Assume one step per episode for simplicity
        return self._get_state(), reward, done


env = Env(num_drones=5, num_tasks=3)
