import torch
import torch.nn as nn
import torch.nn.functional as F


class GNN(nn.Module):
    def __init__(self, drone_feature_dim, task_feature_dim, output_dim):
        super(GNN, self).__init__()
        self.drone_fc = nn.Linear(drone_feature_dim, 32)
        self.task_fc = nn.Linear(task_feature_dim, 32)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, drone_features, task_features):
        drone_embed = F.relu(self.drone_fc(drone_features))
        task_embed = F.relu(self.task_fc(task_features))

        combined = torch.cat([drone_embed, task_embed], dim=1)

        x = F.relu(self.fc1(combined))
        x = self.fc2(x)

        return x


model = GNN(drone_feature_dim=5, task_feature_dim=3, output_dim=1)
