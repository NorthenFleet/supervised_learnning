import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from env import SampleGenerator, DataPreprocessor
from network import DecisionNetwork
from torch.utils.data import DataLoader
from model_manager import ModelManager

class Train:
    def __init__(self, env_config, network_config, training_config):
        self.data_preprocessor = DataPreprocessor(
            env_config["max_entities"], env_config["max_tasks"], env_config["entity_dim"], env_config["task_dim"])
        self.dataset = SampleGenerator(
            training_config["num_samples"], self.data_preprocessor)
        self.dataloader = DataLoader(
            self.dataset, batch_size=training_config["batch_size"], shuffle=True)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = DecisionNetwork(env_config["entity_dim"], network_config["entity_num_heads"], env_config["task_dim"],
                                     network_config["task_num_heads"], network_config["hidden_dim"], network_config["num_layers"], network_config["mlp_hidden_dim"], env_config["max_entities"], network_config["output_dim"])
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=training_config["lr"])
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.num_epochs = training_config["num_epochs"]
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=training_config.get("log_dir", "./runs"))

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            for entities, tasks, entity_mask, task_mask, (task_assignments, task_scores) in self.dataloader:
                entities, tasks, entity_mask, task_mask = entities.to(self.device), tasks.to(
                    self.device), entity_mask.to(self.device), task_mask.to(self.device)
                task_assignments = task_assignments.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(entities, tasks, entity_mask, task_mask)

                # Ensure outputs and task_assignments dimensions match
                assert outputs.shape[0] == task_assignments.shape[0], "Output and task assignments dimension mismatch"

                # Filter out invalid task assignments (-1)
                valid_mask = task_assignments != -1
                valid_task_assignments = task_assignments[valid_mask]
                valid_outputs = outputs[valid_mask]

                # Calculate the loss for each platform-task assignment
                loss = self.criterion(valid_outputs, valid_task_assignments)

                loss.backward()
                self.optimizer.step()

                total_loss += loss
                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            self.scheduler.step()

            # Log the average loss to TensorBoard
            self.writer.add_scalar('Loss/train', avg_loss, epoch)

            print(
                f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss}")

        # Log the model graph (structure) to TensorBoard
        dummy_entities = torch.zeros((self.training_config["batch_size"], self.env_config["max_entities"], self.env_config["entity_dim"])).to(self.device)
        dummy_tasks = torch.zeros((self.training_config["batch_size"], self.env_config["max_tasks"], self.env_config["task_dim"])).to(self.device)
        dummy_entity_mask = torch.ones((self.training_config["batch_size"], self.env_config["max_entities"])).to(self.device)
        dummy_task_mask = torch.ones((self.training_config["batch_size"], self.env_config["max_tasks"])).to(self.device)

        self.writer.add_graph(self.model, (dummy_entities, dummy_tasks, dummy_entity_mask, dummy_task_mask))

        self.writer.close()

    def save_model(self, path):
        ModelManager.save_model(self.model, path)

    def load_model(self, path):
        ModelManager.load_model(self.model, path, self.device)


if __name__ == "__main__":
    env_config = {
        "max_entities": 10,
        "max_tasks": 5,
        "entity_dim": 6,  # Platform position (x, y), range, speed, detection range, endurance
        "task_dim": 4    # Task priority, task position (x, y), task type
    }

    network_config = {
        "entity_num_heads": 2,
        "task_num_heads": 2,
        "hidden_dim": 64,
        "num_layers": 2,
        "mlp_hidden_dim": 128,
        "output_dim": 5     # max_tasks
    }

    training_config = {
        "num_samples": 1000,
        "batch_size": 32,
        "num_epochs": 50,
        "lr": 0.001,
        "log_dir": "./runs"
    }

    trainer = Train(env_config, network_config, training_config)
    trainer.train()
    trainer.save_model("best_model.pth")