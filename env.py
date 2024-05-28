import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
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

        targets = self.__getreward__(entities, tasks)  # 计算最佳任务

        return padded_entities, padded_tasks, entity_mask, task_mask, targets

    def __getreward__(self, entities, tasks):
        num_entities = len(entities)
        num_tasks = len(tasks)

        task_assignments = [-1] * num_entities
        task_scores = np.zeros(num_entities)

        task_assigned = [False] * num_tasks

        for i, entity in enumerate(entities):
            entity_position = entity[:2]  # 平台位置 (x, y)
            entity_speed = entity[3]  # 平台速度

            task_distances = []
            for idx, task in enumerate(tasks):
                task_priority = task[0]
                task_position = task[1:3]  # 任务位置 (x, y)
                distance = np.linalg.norm(entity_position - task_position)
                arrival_time = distance / entity_speed
                task_distances.append((task_priority, arrival_time, idx))

            # 按任务优先级和到达时间排序
            task_distances.sort(key=lambda x: (x[0], x[1]))

            for task in task_distances:
                task_idx = task[2]
                if not task_assigned[task_idx]:
                    task_assignments[i] = task_idx
                    task_scores[i] = task[0] / (task[1] + 1e-5)  # 任务优先级 / 到达时间
                    task_assigned[task_idx] = True
                    break

        # 确保所有任务至少被执行一次
        unassigned_tasks = [idx for idx, assigned in enumerate(task_assigned) if not assigned]
        for task_idx in unassigned_tasks:
            best_entity = None
            best_score = float('-inf')
            for i, entity in enumerate(entities):
                if task_assignments[i] == -1:
                    entity_position = entity[:2]  # 平台位置 (x, y)
                    entity_speed = entity[3]  # 平台速度
                    task_position = tasks[task_idx][1:3]  # 任务位置 (x, y)
                    distance = np.linalg.norm(entity_position - task_position)
                    arrival_time = distance / entity_speed
                    score = tasks[task_idx][0] / (arrival_time + 1e-5)  # 任务优先级 / 到达时间
                    if score > best_score:
                        best_score = score
                        best_entity = i
            
            if best_entity is not None:
                task_assignments[best_entity] = task_idx
                task_scores[best_entity] = best_score
            else:
                # 随机选择一个实体分配任务，以确保任务被执行
                random_entity = np.random.choice(range(num_entities))
                task_assignments[random_entity] = task_idx
                task_scores[random_entity] = tasks[task_idx][0] / (1 + 1e-5)  # 随机选择实体分配任务

        return task_assignments, task_scores


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
