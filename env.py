import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import numpy as np


class SampleGenerator(Dataset):
    def __init__(self, num_samples, data_preprocessor):
        self.num_samples = num_samples
        self.data_preprocessor = data_preprocessor

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        num_entities = np.random.randint(
            1, self.data_preprocessor.max_entities + 1)
        entities = np.zeros((num_entities, self.data_preprocessor.entity_dim))
        for i in range(num_entities):
            x = np.random.uniform(0, 100)                 # 平台位置 x
            y = np.random.uniform(0, 100)                 # 平台位置 y
            range_ = np.random.uniform(50, 500)           # 航程
            speed = np.random.uniform(10, 30)             # 速度
            detection_range = np.random.uniform(10, 100)  # 探测距离
            endurance = np.random.uniform(1, 10)          # 可持续时长
            entities[i] = [x, y, range_, speed, detection_range, endurance]

        num_tasks = np.random.randint(1, self.data_preprocessor.max_tasks + 1)
        tasks = np.zeros((num_tasks, self.data_preprocessor.task_dim))
        for j in range(num_tasks):
            priority = np.random.randint(1, 4)            # 任务优先级
            x = np.random.uniform(0, 100)                 # 任务位置 x
            y = np.random.uniform(0, 100)                 # 任务位置 y
            # 任务类型 (侦察=0, 打击=1, 支援=2)
            task_type = np.random.randint(0, 3)
            tasks[j] = [priority, x, y, task_type]

        padded_entities, padded_tasks, entity_mask, task_mask = self.data_preprocessor.pad_and_mask(
            entities, tasks)

        target = self.__getreward__(entities, tasks)  # 计算最佳任务

        return padded_entities, padded_tasks, entity_mask, task_mask, target

    def __getreward__(self, entities, tasks):
        first_entity = entities[0]  # 获取第一个算子
        entity_position = first_entity[:2]  # 第一个算子的位置 (x, y)
        entity_speed = first_entity[3]  # 第一个算子的速度

        task_distances = []
        for idx, task in enumerate(tasks):
            task_priority = task[0]
            task_position = task[1:3]  # 任务位置 (x, y)
            distance = np.linalg.norm(entity_position - task_position)
            arrival_time = distance / entity_speed
            task_distances.append((task_priority, arrival_time, idx))

        # 按任务优先级和到达时间排序
        task_distances.sort(key=lambda x: (x[0], x[1]))

        # 返回第一个算子最佳任务的序号
        return task_distances[0][2]


class DataPreprocessor:
    def __init__(self, max_entities, max_tasks, entity_dim, task_dim):
        self.max_entities = max_entities
        self.max_tasks = max_tasks
        self.entity_dim = entity_dim
        self.task_dim = task_dim

    def pad_and_mask(self, entities, tasks):
        entities_padded = np.zeros((self.max_entities, self.entity_dim))
        tasks_padded = np.zeros((self.max_tasks, self.task_dim))

        num_entities = entities.shape[0]
        num_tasks = tasks.shape[0]

        entities_padded[:num_entities] = entities
        tasks_padded[:num_tasks] = tasks

        entity_mask = np.zeros(self.max_entities)
        task_mask = np.zeros(self.max_tasks)
        entity_mask[:num_entities] = 1
        task_mask[:num_tasks] = 1

        return (
            torch.tensor(entities_padded, dtype=torch.float32),
            torch.tensor(tasks_padded, dtype=torch.float32),
            torch.tensor(entity_mask, dtype=torch.float32),
            torch.tensor(task_mask, dtype=torch.float32)
        )
