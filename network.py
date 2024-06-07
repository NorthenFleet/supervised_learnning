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
    def __init__(self, entity_input_dim, task_input_dim,
                 transfer_dim, entity_num_heads, task_num_heads,
                 hidden_dim, num_layers, mlp_hidden_dim,
                 entity_heads, output_dim, use_transformer):
        super(DecisionNetworkMultiHead, self).__init__()
        self.use_transformer = use_transformer
        self.entity_norm = nn.LayerNorm(entity_input_dim)
        self.task_norm = nn.LayerNorm(task_input_dim)

        self.entity_embedding = nn.Linear(entity_input_dim, transfer_dim)
        self.task_embedding = nn.Linear(task_input_dim, transfer_dim)

        # 定义全连接层
        self.entity_fc_layers = self._build_fc_layers(
            transfer_dim, hidden_dim, num_layers)
        self.task_fc_layers = self._build_fc_layers(
            transfer_dim, hidden_dim, num_layers)
        
        self.entity_encoder = TransformerEncoder(
            transfer_dim, entity_num_heads, hidden_dim, num_layers)
        self.task_encoder = TransformerEncoder(
            transfer_dim, task_num_heads, hidden_dim, num_layers)

        self.entity_bn = nn.BatchNorm1d(transfer_dim)
        self.task_bn = nn.BatchNorm1d(transfer_dim)

        self.combination_layer = nn.Linear(2 * transfer_dim, transfer_dim)
        self.hidden_layer = nn.Linear(transfer_dim, transfer_dim)
        self.activation = nn.ReLU()

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(transfer_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(mlp_hidden_dim, output_dim)
            ) for _ in range(entity_heads)
        ])

    def _build_fc_layers(self, input_dim, hidden_dim, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        return nn.Sequential(*layers)

    def forward(self, entities, tasks, entity_mask, task_mask):
        # normalization
        entities = self.entity_norm(entities)
        tasks = self.task_norm(tasks)

        # Embedding and permute for transformers
        entities = self.entity_embedding(entities)
        tasks = self.task_embedding(tasks)

        if self.use_transformer == True:
            entities = entities.permute(1, 0, 2)
            tasks = tasks.permute(1, 0, 2)
            # Encoding
            encoded_entities = self.entity_encoder(
                entities, src_key_padding_mask=entity_mask.bool()).max(dim=0)[0]
            encoded_tasks = self.task_encoder(
                tasks, src_key_padding_mask=task_mask.bool()).max(dim=0)[0]

        else:
            # Embedding and normalization
            entities = self.entity_bn(self.entity_embedding(entities))
            tasks = self.task_bn(self.task_embedding(tasks))

            # Fully connected layers
            entities = self.entity_fc_layers(
                entities.permute(0, 2, 1)).permute(0, 2, 1)
            tasks = self.task_fc_layers(tasks.permute(0, 2, 1)).permute(0, 2, 1)

            # Max pooling
            encoded_entities = entities.max(dim=1)[0]
            encoded_tasks = tasks.max(dim=1)[0]

        # Combine entity and task encodings
        combined_output = torch.cat((encoded_entities, encoded_tasks), dim=-1)
        combined_output = self.combination_layer(combined_output)
        combined_output = self.activation(combined_output)
        # combined_output = self.hidden_layer(combined_output)
        # combined_output = self.activation(combined_output)

        # Multi-head outputs
        outputs = []
        for i in range(len(self.heads)):
            output = self.heads[i](combined_output)

            # 处理无效行
            if torch.isinf(output).all(dim=-1).any():
                output[torch.isinf(output).all(dim=-1)] = 0

            output = F.softmax(output, dim=-1)

            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)

        return outputs

    def predict(self, entities, tasks, entity_mask, task_mask):
        outputs = self.forward(entities, tasks, entity_mask, task_mask)
        return torch.argmax(outputs, dim=-1)  # Return indices for prediction
