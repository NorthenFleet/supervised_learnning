import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


class Env:
    def __init__(self, num_drones, num_tasks):
        self.num_drones = num_drones
        self.num_tasks = num_tasks
        self.reset()

    def reset(self):
        self.drones = [
            {
                "id": i,
                "position": np.random.rand(2),
                "range": np.random.rand() * 100,
                "speed": np.random.rand() * 10,
                "endurance": np.random.rand() * 100
            }
            for i in range(self.num_drones)
        ]
        self.tasks = [
            {
                "id": i,
                "position": np.random.rand(2),
                "priority": random.randint(1, 10)
            }
            for i in range(self.num_tasks)
        ]
        return self._get_state()

    def _get_state(self):
        drone_features = torch.tensor(
            [[d["position"][0], d["position"][1], d["range"],
                d["speed"], d["endurance"]] for d in self.drones],
            dtype=torch.float
        )
        task_features = torch.tensor(
            [[t["position"][0], t["position"][1], t["priority"]]
                for t in self.tasks],
            dtype=torch.float
        )
        return drone_features, task_features

    def get_action(self):
        UAVs = self.drones[:]
        Targets = self.tasks[:]
        T_list = []
        for i in Targets:
            if T_list == []:
                T_list.append(i)
            else:
                for j in range(len(T_list)):
                    flag = True
                    if flag:
                        if T_list[j]['priority'] < i['priority']:
                            T_list.insert(j, i)
                            flag = False
                            break
                    if flag and j == len(T_list)-1:
                        T_list.append(i)

        mission_list = []

        for i in T_list:
            timelist = []
            for j in UAVs:
                timelist.append(
                    [distance(i['position'], j['position']) / j['speed'], j])
            timelist.sort()
            for m in timelist:
                if m[0] > m[1]['range'] or m[0] > m[1]['endurance']:
                    continue
                else:
                    mission_list.append(m[1]['id'])
                    print('最优决策是:编组编号:%d,任务编号:%d,任务开始时间:%f' %
                          (m[1]['id'], i['id'], m[0]))

                    idx_uav = UAVs.index(m[1])
                    idx_target = Targets.index(i)

                    UAVs = UAVs[:idx_uav] + UAVs[idx_uav+1:]
                    Targets = Targets[:idx_target] + Targets[idx_target+1:]
                    break

        return mission_list

    def step(self, action):
        reward = random.random()  # Placeholder for actual reward calculation logic
        done = True  # Assuming one step per episode for simplicity
        return self._get_state(), reward, done


env = Env(num_drones=5, num_tasks=3)
