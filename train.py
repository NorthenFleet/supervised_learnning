import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
            self.dataset, batch_size=training_config["batch_size"], shuffle=True, collate_fn=self.collate_fn)

        self.val_dataset = SampleGenerator(
            training_config["num_samples"] // 10, self.data_preprocessor)  # 假设验证集为训练集的10%
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=training_config["batch_size"], shuffle=False, collate_fn=self.collate_fn)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = DecisionNetworkMultiHead(env_config["entity_dim"], env_config["task_dim"], network_config["transfer_dim"], network_config["entity_num_heads"],
                                              network_config["task_num_heads"], network_config["hidden_dim"], network_config["num_layers"], network_config["mlp_hidden_dim"], env_config["max_entities"], network_config["output_dim"])
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=training_config["lr"])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5)

        self.num_epochs = training_config["num_epochs"]
        self.patience = training_config.get("patience", 10)  # 默认早停耐心值为10个epoch
        self.logger = TensorBoardLogger()

        # Log the model graph (structure) to TensorBoard outside of training loop
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

    def collate_fn(self, batch):
        entities, tasks, entity_mask, task_mask, task_assignments = zip(*batch)
        entities = torch.stack(entities)
        tasks = torch.stack(tasks)
        entity_mask = torch.stack(entity_mask)
        task_mask = torch.stack(task_mask)

        # 修正 task_assignments 以确保每个元素都是整数张量并过滤掉 -1
        task_assignments = [torch.tensor(
            ta, dtype=torch.long) for ta in task_assignments]
        for ta in task_assignments:
            ta[ta == -1] = 0  # 将所有 -1 转换为 0
        task_assignments = torch.stack(task_assignments)
        return entities, tasks, entity_mask, task_mask, task_assignments

    def validate(self):
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for entities, tasks, entity_mask, task_mask, targets in self.val_dataloader:
                entities, tasks, entity_mask, task_mask, targets = entities.to(self.device), tasks.to(
                    self.device), entity_mask.to(self.device), task_mask.to(self.device), targets.to(self.device)

                outputs = self.model(
                    entities, tasks, entity_mask, task_mask)

                loss = 0
                for i, output in enumerate(outputs):
                    loss += self.criterion(output, targets[:, i])

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(self.val_dataloader)
        return avg_val_loss

    def early_stopping_check(self, avg_val_loss, best_val_loss, patience_counter, path):
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            self.save_model(path)
        else:
            patience_counter += 1
            if patience_counter >= self.patience:
                print("Early stopping triggered")
                return best_val_loss, patience_counter, True
        return best_val_loss, patience_counter, False

    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
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
                loss = self.criterion(outputs, task_assignments)

                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.dataloader)

            # 验证集评估
            avg_val_loss = self.validate()

            # 调整学习率
            self.scheduler.step(avg_val_loss)

            # Log the average losses to TensorBoard
            self.logger.log_scalar('Loss/train', avg_train_loss, epoch)
            self.logger.log_scalar('Loss/val', avg_val_loss, epoch)

            print(
                f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

            # 早停逻辑
            best_val_loss, patience_counter, stop = self.early_stopping_check(
                avg_val_loss, best_val_loss, patience_counter, "best_model.pth")
            if stop:
                break

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
        "output_dim": 5,     # max_tasks
        "transfer_dim": 10
    }

    training_config = {
        "num_samples": 1000,
        "batch_size": 32,
        "lr": 0.001,
        "num_epochs": 50,
        "patience": 10
    }

    trainer = Train(env_config, network_config, training_config)

    # 加载现有的模型
    model_path = "best_model.pth"
    try:
        trainer.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(
            f"No existing model found at {model_path}. Starting training from scratch.")

    trainer.train()
    trainer.save_model("best_model.pth")
