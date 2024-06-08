import torch
import torch.nn as nn

# 输入维度
entity_dim = 6
task_dim = 4
embed_dim = 10  # 嵌入后的维度

# 定义模型


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.norm_entities = nn.BatchNorm1d(entity_dim)
        self.norm_tasks = nn.BatchNorm1d(task_dim)
        self.embed_entities = nn.Linear(entity_dim, embed_dim)
        self.embed_tasks = nn.Linear(task_dim, embed_dim)

    def forward(self, entities, tasks):
        # 归一化和嵌入
        entities_bn = self.norm_entities(
            entities.view(-1, entity_dim)).view(entities.shape)
        tasks_bn = self.norm_tasks(tasks.view(-1, task_dim)).view(tasks.shape)
        entities_embedded = self.embed_entities(entities_bn)
        tasks_embedded = self.embed_tasks(tasks_bn)

        # 拼接嵌入后的数据
        combined = torch.cat([entities_embedded, tasks_embedded], dim=1)
        return combined


# 示例批量数据
entities = torch.randn(32, 10, 6)  # 32批，每批10个实体，每个实体6个属性
tasks = torch.randn(32, 7, 4)      # 32批，每批7个任务，每个任务4个属性

model = MyModel()
output = model(entities, tasks)
print(output.shape)
