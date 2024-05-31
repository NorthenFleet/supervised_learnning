import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers)

    def forward(self, src, src_mask):
        src_key_padding_mask = (src_mask == 0)
        return self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)


class DecisionNetwork(nn.Module):
    def __init__(self, entity_input_dim, entity_num_heads, task_input_dim, task_num_heads, hidden_dim, num_layers, mlp_hidden_dim, max_entities, output_dim):
        super(DecisionNetwork, self).__init__()
        self.entity_encoder = TransformerEncoder(
            entity_input_dim, entity_num_heads, hidden_dim, num_layers)
        self.task_encoder = TransformerEncoder(
            task_input_dim, task_num_heads, hidden_dim, num_layers)

        self.max_entities = max_entities

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

        outputs = []
        for _ in range(self.max_entities):
            output = self.mlp(combined)
            outputs.append(output)

        return torch.stack(outputs, dim=1)


class DecisionNetworkMultiHead(nn.Module):
    def __init__(self, entity_input_dim, entity_num_heads, task_input_dim, task_num_heads, hidden_dim, num_layers, mlp_hidden_dim, max_entities, output_dim):
        super(DecisionNetworkMultiHead, self).__init__()
        self.max_entities = max_entities
        self.entity_encoder = TransformerEncoder(
            entity_input_dim, entity_num_heads, hidden_dim, num_layers)
        self.task_encoder = TransformerEncoder(
            task_input_dim, task_num_heads, hidden_dim, num_layers)

        # 多头输出，每个头部对应一个平台的任务分配
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(entity_input_dim + task_input_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(mlp_hidden_dim, output_dim),
                nn.Softmax(dim=-1)  # 输出任务编号的概率分布
            ) for _ in range(max_entities)
        ])

    def forward(self, entities, tasks, entity_mask, task_mask):
        entities = entities.permute(1, 0, 2)
        tasks = tasks.permute(1, 0, 2)

        encoded_entities = self.entity_encoder(
            entities, entity_mask).mean(dim=0)
        encoded_tasks = self.task_encoder(tasks, task_mask).mean(dim=0)

        outputs = []
        for i in range(self.max_entities):
            # expand encoded_entities[i] to match the first dimension of encoded_tasks
            combined = torch.cat((encoded_entities[i].unsqueeze(0).expand(
                encoded_tasks.size(0), -1), encoded_tasks), dim=-1)
            output = self.heads[i](combined)
            outputs.append(output)

        return torch.stack(outputs, dim=1)
