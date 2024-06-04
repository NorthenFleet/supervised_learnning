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

    def forward(self, src, src_key_padding_mask):
        return self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)


class DecisionNetworkMultiHead(nn.Module):
    def __init__(self, entity_input_dim, task_input_dim, transfer_dim, entity_num_heads, task_num_heads, hidden_dim, num_layers, mlp_hidden_dim, max_entities, output_dim):
        super(DecisionNetworkMultiHead, self).__init__()
        self.entity_embedding = nn.Linear(entity_input_dim, transfer_dim)
        self.task_embedding = nn.Linear(task_input_dim, transfer_dim)
        self.entity_encoder = TransformerEncoder(
            transfer_dim, entity_num_heads, hidden_dim, num_layers)
        self.task_encoder = TransformerEncoder(
            transfer_dim, task_num_heads, hidden_dim, num_layers)

        self.combination_layer = nn.Linear(2 * transfer_dim, transfer_dim)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(transfer_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(mlp_hidden_dim, output_dim + 1)  # 输出维度增加1
            ) for _ in range(max_entities)
        ])

    def forward(self, entities, tasks, entity_mask, task_mask):
        '''
        # 检查并处理 entity_mask 和 task_mask 中全为 1 的情况
        if torch.any(entity_mask.sum(dim=1) == entity_mask.size(1)):
            entity_mask[entity_mask.sum(dim=1) == entity_mask.size(1)] = 0
        if torch.any(task_mask.sum(dim=1) == task_mask.size(1)):
            task_mask[task_mask.sum(dim=1) == task_mask.size(1)] = 0
        '''
        # Embedding and permute for transformers
        entities = self.entity_embedding(entities).permute(1, 0, 2)
        tasks = self.task_embedding(tasks).permute(1, 0, 2)

        # Encoding
        encoded_entities = self.entity_encoder(
            entities, src_key_padding_mask=entity_mask.bool()).max(dim=0)[0]
        if torch.isnan(encoded_entities).any():
            print("NaN detected in encoded_entities")
            return None

        encoded_tasks = self.task_encoder(
            tasks, src_key_padding_mask=task_mask.bool()).max(dim=0)[0]
        if torch.isnan(encoded_tasks).any():
            print("NaN detected in encoded_tasks")
            return None

        # Combine entity and task encodings
        combined_output = torch.cat((encoded_entities, encoded_tasks), dim=-1)
        combined_output = self.combination_layer(combined_output)
        combined_output = F.relu(combined_output)

        if torch.isnan(combined_output).any():
            print("NaN detected in combined_output")
            return None

        # Multi-head outputs
        outputs = []
        for i in range(len(self.heads)):
            output = self.heads[i](combined_output)
            # # Apply mask before softmax
            # output = output.masked_fill(~task_mask.bool(), float('-inf'))

            # 处理无效行
            if torch.isinf(output).all(dim=-1).any():
                output[torch.isinf(output).all(dim=-1)] = 0

            output = F.softmax(output, dim=-1)
            if torch.isnan(output).any():
                print("NaN detected in outputs")
                return None  # 或者采取其他适当的措施
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        if torch.isnan(outputs).any():
            print("NaN detected in outputs")
            return None

        return outputs

    def predict(self, entities, tasks, entity_mask, task_mask):
        outputs = self.forward(entities, tasks, entity_mask, task_mask)
        return torch.argmax(outputs, dim=-1)  # Return indices for prediction


