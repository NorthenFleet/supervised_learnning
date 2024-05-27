import ray
import torch
import torch.nn as nn
from env import SampleGenerator, DataPreprocessor
from network import DecisionNetwork
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from model_manager import ModelManager
from ray import tune
from ray.tune.schedulers import ASHAScheduler


class TrainModel:
    def __init__(self, config):
        self.config = config
        self.data_preprocessor = DataPreprocessor(
            config["max_entities"], config["max_tasks"], config["entity_dim"], config["task_dim"])
        self.dataset = SampleGenerator(
            config["num_samples"], self.data_preprocessor)

        # 将数据集分为训练集和验证集
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=config["batch_size"], shuffle=True)
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=config["batch_size"], shuffle=False)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = DecisionNetwork(config["entity_dim"], config["entity_num_heads"], config["task_dim"],
                                     config["task_num_heads"], config["hidden_dim"], config["num_layers"], config["mlp_hidden_dim"], config["max_entities"], config["output_dim"])
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config["lr"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.1)

    def train(self):
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(self.config["num_epochs"]):
            self.model.train()
            total_train_loss = 0.0
            for entities, tasks, entity_mask, task_mask, targets in self.train_dataloader:
                entities, tasks, entity_mask, task_mask, targets = entities.to(self.device), tasks.to(
                    self.device), entity_mask.to(self.device), task_mask.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(entities, tasks, entity_mask, task_mask)
                
                loss = 0
                for i, output in enumerate(outputs):
                    loss += self.criterion(output, targets[:, i])
                
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(self.train_dataloader)

            # 验证集评估
            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for entities, tasks, entity_mask, task_mask, targets in self.val_dataloader:
                    entities, tasks, entity_mask, task_mask, targets = entities.to(self.device), tasks.to(
                        self.device), entity_mask.to(self.device), task_mask.to(self.device), targets.to(self.device)

                    outputs = self.model(entities, tasks, entity_mask, task_mask)
                    
                    loss = 0
                    for i, output in enumerate(outputs):
                        loss += self.criterion(output, targets[:, i])

                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(self.val_dataloader)
            self.scheduler.step(avg_val_loss)

            tune.report(loss=avg_val_loss)
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

            # 早停逻辑
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_model("best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

    def save_model(self, path):
        ModelManager.save_model(self.model, path)

    def load_model(self, path):
        ModelManager.load_model(self.model, path, self.device)

    def train_with_ray(self):
        analysis = tune.run(
            self.train_model_with_ray,
            config=self.config,
            num_samples=10,
            scheduler=ASHAScheduler(metric="loss", mode="min"),
            resources_per_trial={"cpu": 1, "gpu": 1}  # 依据实际硬件情况调整
        )

        print("Best config: ", analysis.best_config)
        self.load_model(analysis.best_checkpoint)

    def train_model_with_ray(self, config):
        trainer = TrainModel(config)
        trainer.train()


if __name__ == "__main__":
    config = {
        "num_samples": 1000,
        "max_entities": 10,
        "max_tasks": 5,
        "entity_dim": 6,  # 平台位置 (x, y), 航程, 速度, 探测距离, 可持续时长
        "task_dim": 4,    # 任务优先级, 任务位置 (x, y), 任务类型
        "batch_size": 32,
        "entity_num_heads": 2,
        "task_num_heads": 2,
        "hidden_dim": 64,
        "num_layers": 2,
        "mlp_hidden_dim": 128,
        "max_entities": 10,  # 与 max_entities 保持一致
        "output_dim": 5,     # max_tasks
        "lr": 0.001,
        "num_epochs": 50
    }

    ray.init()
    trainer = TrainModel(config)
    
    # 加载现有的模型
    model_path = "best_model.pth"
    try:
        trainer.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"No existing model found at {model_path}. Starting training from scratch.")

    trainer.train_with_ray()
    trainer.save_model("best_model.pth")

