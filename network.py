import torch
import torch.nn as nn
import torch.nn.functional as F


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
