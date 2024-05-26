import ray
import torch
import torch.nn as nn


class TrainModel:
    def __init__(self, config):
        self.config = config
        self.data_preprocessor = DataPreprocessor(
            config["max_entities"], config["max_tasks"], config["entity_dim"], config["task_dim"])
        self.dataset = SampleGenerator(
            config["num_samples"], self.data_preprocessor)
        self.dataloader = DataLoader(
            self.dataset, batch_size=config["batch_size"], shuffle=True)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = DecisionNetwork(config["entity_dim"], config["entity_num_heads"], config["task_dim"],
                                     config["task_num_heads"], config["hidden_dim"], config["num_layers"], config["mlp_hidden_dim"], config["output_dim"])
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config["lr"])

    def train(self):
        for epoch in range(self.config["num_epochs"]):
            self.model.train()
            total_loss = 0.0
            for entities, tasks, entity_mask, task_mask, targets in self.dataloader:
                entities, tasks, entity_mask, task_mask, targets = entities.to(self.device), tasks.to(
                    self.device), entity_mask.to(self.device), task_mask.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(entities, tasks, entity_mask, task_mask)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(
                f"Epoch {epoch + 1}/{self.config['num_epochs']}, Loss: {total_loss / len(self.dataloader)}")

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model.load_model(path)

    def train_with_ray(self):
        def train_model_with_ray(config):
            trainer = TrainModel(config)
            trainer.train()
            tune.report(loss=total_loss / len(self.dataloader))

        analysis = tune.run(
            train_model_with_ray,
            config=self.config,
            num_samples=10,
            scheduler=ASHAScheduler(metric="loss", mode="min")
        )

        print("Best config: ", analysis.best_config)
        self.load_model(analysis.best_checkpoint)


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
        "output_dim": 5,
        "lr": 0.001,
        "num_epochs": 50
    }

    ray.init()
    trainer = TrainModel(config)
    trainer.train_with_ray()
    trainer.save_model("best_model.pth")
