import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers)

    def forward(self, src, src_mask):
        src_key_padding_mask = (src_mask == 0)
        return self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)


class DecisionNetwork(nn.Module):
    def __init__(self, entity_input_dim, entity_num_heads, task_input_dim, task_num_heads, hidden_dim, num_layers, mlp_hidden_dim, output_dim):
        super(DecisionNetwork, self).__init__()

        self.entity_encoder = TransformerEncoder(
            entity_input_dim, entity_num_heads, hidden_dim, num_layers)
        self.task_encoder = TransformerEncoder(
            task_input_dim, task_num_heads, hidden_dim, num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(entity_input_dim+task_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(mlp_hidden_dim, output_dim)
        )

    def forward(self, entities, tasks, entity_mask, task_mask):
        entities = entities.permute(1, 0, 2)
        tasks = tasks.permute(1, 0, 2)

        encoded_entities = self.entity_encoder(
            entities, entity_mask).mean(dim=0)
        encoded_tasks = self.task_encoder(tasks, task_mask).mean(dim=0)

        combined = torch.cat((encoded_entities, encoded_tasks), dim=1)
        output = self.mlp(combined)
        return F.softmax(output, dim=-1)


def train_model(model, dataloader, num_epochs, criterion, optimizer, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for entities, tasks, entity_mask, task_mask, targets in dataloader:
            entities, tasks, entity_mask, task_mask, targets = entities.to(device), tasks.to(
                device), entity_mask.to(device), task_mask.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(entities, tasks, entity_mask, task_mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")


if __name__ == "__main__":
    num_samples = 1000
    max_entities = 10
    max_tasks = 5
    entity_dim = 6  # 平台位置 (x, y), 航程, 速度, 探测距离, 可持续时长
    task_dim = 4    # 任务优先级, 任务位置 (x, y), 任务类型
    batch_size = 32
    entity_num_heads = 2
    task_num_heads = 2
    hidden_dim = 64
    num_layers = 2
    mlp_hidden_dim = 128
    output_dim = max_tasks
    num_epochs = 10

    data_preprocessor = DataPreprocessor(
        max_entities, max_tasks, entity_dim, task_dim)
    dataset = SampleGenerator(num_samples, data_preprocessor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DecisionNetwork(entity_dim, entity_num_heads, task_dim,
                            task_num_heads, hidden_dim, num_layers, mlp_hidden_dim, output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, num_epochs, criterion, optimizer, device)
