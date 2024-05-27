import torch
import torch.nn as nn
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from env import SampleGenerator, DataPreprocessor
from network import DecisionNetwork
from torch.utils.data import DataLoader
from model_manager import ModelManager


class TrainModel:
    def __init__(self, env_config, network_config, training_config):
        self.env_config = env_config
        self.network_config = network_config
        self.training_config = training_config
        
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
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=training_config["lr"])

    def train(self):
        for epoch in range(self.training_config["num_epochs"]):
            self.model.train()
            total_loss = 0.0
            for entities, tasks, entity_mask, task_mask, targets in self.dataloader:
                entities, tasks, entity_mask, task_mask, targets = entities.to(self.device), tasks.to(
                    self.device), entity_mask.to(self.device), task_mask.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(entities, tasks, entity_mask, task_mask)
                
                # 计算每个平台对应任务的损失
                loss = 0
                for i, output in enumerate(outputs):
                    loss += self.criterion(output, targets[:, i])
                
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            tune.report(loss=total_loss / len(self.dataloader))
            print(f"Epoch {epoch + 1}/{self.training_config['num_epochs']}, Loss: {total_loss / len(self.dataloader)}")

    def save_model(self, path):
        ModelManager.save_model(self.model, path)

    def load_model(self, path):
        ModelManager.load_model(self.model, path, self.device)

    def train_with_ray(self):
        analysis = tune.run(
            self.train_model_with_ray,
            config={
                "env_config": self.env_config,
                "network_config": self.network_config,
                "training_config": self.training_config
            },
            num_samples=10,  # 这是并行试验的数量
            scheduler=ASHAScheduler(metric="loss", mode="min"),
            resources_per_trial={"cpu": 1, "gpu": 1}  # 依据实际硬件情况调整
        )

        print("Best config: ", analysis.best_config)
        self.load_model(analysis.best_checkpoint)

    def train_model_with_ray(self, config):
        trainer = TrainModel(config["env_config"], config["network_config"], config["training_config"])
        trainer.train()


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

    ray.init()
    trainer = TrainModel(env_config, network_config, training_config)
    
    # 加载现有的模型
    model_path = "best_model.pth"
    try:
        trainer.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"No existing model found at {model_path}. Starting training from scratch.")

    trainer.train_with_ray()
    trainer.save_model("best_model.pth")
