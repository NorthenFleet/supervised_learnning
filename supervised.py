import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SampleGenerator(Dataset):
    def __init__(self, num_samples, max_entities, max_tasks, entity_dim, task_dim):
        self.num_samples = num_samples
        self.max_entities = max_entities
        self.max_tasks = max_tasks
        self.entity_dim = entity_dim
        self.task_dim = task_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        num_entities = np.random.randint(1, self.max_entities + 1)
        num_tasks = np.random.randint(1, self.max_tasks + 1)

        entities = np.random.randn(num_entities, self.entity_dim)
        tasks = np.random.randn(num_tasks, self.task_dim)

        # Padding entities and tasks to max length
        entities_padded = np.zeros((self.max_entities, self.entity_dim))
        tasks_padded = np.zeros((self.max_tasks, self.task_dim))

        entities_padded[:num_entities] = entities
        tasks_padded[:num_tasks] = tasks

        # Mask for padding (1 for real, 0 for padding)
        entity_mask = np.zeros(self.max_entities)
        task_mask = np.zeros(self.max_tasks)
        entity_mask[:num_entities] = 1
        task_mask[:num_tasks] = 1

        # Example: random task assignment for demonstration purposes
        target = np.random.randint(0, num_tasks)

        return (
            torch.tensor(entities_padded, dtype=torch.float32),
            torch.tensor(tasks_padded, dtype=torch.float32),
            torch.tensor(entity_mask, dtype=torch.float32),
            torch.tensor(task_mask, dtype=torch.float32),
            target
        )


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
    def __init__(self, entity_input_dim, task_input_dim, num_heads, hidden_dim, num_layers, mlp_hidden_dim, output_dim):
        super(DecisionNetwork, self).__init__()
        self.entity_encoder = TransformerEncoder(
            entity_input_dim, num_heads, hidden_dim, num_layers)
        self.task_encoder = TransformerEncoder(
            task_input_dim, num_heads, hidden_dim, num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_hidden_dim),
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
    entity_dim = 8
    task_dim = 6
    batch_size = 32
    num_heads = 4
    hidden_dim = 64
    num_layers = 2
    mlp_hidden_dim = 128
    output_dim = max_tasks
    num_epochs = 10

    dataset = SampleGenerator(
        num_samples, max_entities, max_tasks, entity_dim, task_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DecisionNetwork(entity_dim, task_dim, num_heads,
                            hidden_dim, num_layers, mlp_hidden_dim, output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, num_epochs, criterion, optimizer, device)
