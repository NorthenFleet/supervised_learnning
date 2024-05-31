import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from env import SampleGenerator, DataPreprocessor
from network import DecisionNetworkMultiHead
from model_manager import ModelManager
from tensorboard_logger import TensorBoardLogger


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

        self.model = DecisionNetworkMultiHead(env_config["entity_dim"], network_config["entity_num_heads"], env_config["task_dim"],
                                              network_config["task_num_heads"], network_config["hidden_dim"], network_config["num_layers"], network_config["mlp_hidden_dim"], env_config["max_entities"], network_config["output_dim"])
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=training_config["lr"])
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)

        self.num_epochs = training_config["num_epochs"]
        self.logger = TensorBoardLogger()

        # Log the model graph (structure) to TensorBoard
        dummy_entities = torch.zeros(
            (training_config["batch_size"], env_config["max_entities"], env_config["entity_dim"])).to(self.device)
        dummy_tasks = torch.zeros(
            (training_config["batch_size"], env_config["max_tasks"], env_config["task_dim"])).to(self.device)
        dummy_entity_mask = torch.ones(
            (training_config["batch_size"], env_config["max_entities"])).to(self.device)
        dummy_task_mask = torch.ones(
            (training_config["batch_size"], env_config["max_tasks"])).to(self.device)

        self.logger.log_graph(
            self.model, (dummy_entities, dummy_tasks, dummy_entity_mask, dummy_task_mask))

    def train(self):
        self.model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            for entities, tasks, entity_mask, task_mask, task_assignments in self.dataloader:
                entities, tasks, entity_mask, task_mask = entities.to(self.device), tasks.to(
                    self.device), entity_mask.to(self.device), task_mask.to(self.device)
                task_assignments = task_assignments.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(entities, tasks, entity_mask, task_mask)

                # 确保outputs和task_assignments的维度匹配
                outputs = torch.stack(outputs, dim=1)
                assert outputs.shape[:-
                                     1] == task_assignments.shape, "输出和任务分配的维度不匹配"

                # 计算每个平台对应任务的损失
                loss = 0
                for i in range(outputs.shape[1]):
                    loss += self.criterion(outputs[:, i],
                                           task_assignments[:, i])

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            self.scheduler.step()
            avg_loss = total_loss / len(self.dataloader)

            # Log the average loss to TensorBoard
            self.logger.log_scalar('Loss/train', avg_loss, epoch)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss}")

        self.logger.close()

    def save_model(self, path):
        ModelManager.save_model(self.model, path)

    def load_model(self, path):
        ModelManager.load_model(self.model, path, self.device)


if __name__ == "__main__":
    env_config = {
        "max_entities": 10,
        "max_tasks": 5,
        "entity_dim": 6,  # 平台位置 (x, y), 航程, 速度, 探测距离, 可持续时长
        "task_dim": 4    # 任务优先级, 任务位置 (x, y), 任务类型
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
        "lr": 0.001,
        "num_epochs": 50
    }

    trainer = Train(env_config, network_config, training_config)
    trainer.train()
    trainer.save_model("best_model.pth")
