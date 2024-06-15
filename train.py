import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from env import SampleGenerator, DataPreprocessor
from network import DecisionNetworkMultiHead
from model_manager import ModelManager
from tensorboard_logger import TensorBoardLogger


class Trainer:
    def __init__(self, env_config, network_config, training_config, data_file=None):
        self.network_config = network_config
        self.data_preprocessor = DataPreprocessor(
            env_config["max_entities"], env_config["max_tasks"], env_config["entity_dim"], env_config["task_dim"])

        self.dataset = SampleGenerator(
            env_config["num_samples"], env_config, self.data_preprocessor)

        self.dataloader = DataLoader(
            self.dataset, batch_size=training_config["batch_size"], shuffle=True, collate_fn=self.collate_fn)

        self.val_dataset = SampleGenerator(
            env_config["num_samples"] // 10, env_config, self.data_preprocessor)
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=training_config["batch_size"], shuffle=False, collate_fn=self.collate_fn)

        torch.cuda.set_device(0)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = DecisionNetworkMultiHead(
            network_config["max_entities"], network_config["max_tasks"],
            network_config["entity_input_dim"], network_config["task_input_dim"], network_config["transfer_dim"],
            network_config["entity_transformer_heads"], network_config["task_transformer_heads"],
            network_config["hidden_dim"], network_config["num_layers"],
            network_config["mlp_hidden_dim"], env_config["max_entities"],
            network_config["output_dim"] +
            1, network_config["use_transformer"],
            network_config["use_head_mask"]
        )  # 增加一个任务编号
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=training_config["lr"])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode=training_config["lr_mode"], factor=training_config["factor"], patience=training_config["patience"])

        self.num_epochs = training_config["num_epochs"]
        self.patience = training_config.get("patience", 10)
        self.epsiode = training_config.get("epsiode", 100)
        self.logger = TensorBoardLogger()

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

        self.model_path = "%s_%s_%s_model.pth" % (
            env_config["max_entities"], env_config["max_tasks"], network_config["use_transformer"])
        self.name = "%s_%s_%s" % (
            env_config["max_entities"], env_config["max_tasks"], network_config["use_transformer"])

        try:
            self.load_model()
            print(f"Successfully loaded model from {self.model_path}")
        except FileNotFoundError:
            print(
                f"No existing model found at {self.model_path}. Starting training from scratch.")

    def collate_fn(self, batch):
        entities, tasks, entity_mask, task_mask, task_assignments = zip(*batch)
        entities = torch.stack(entities)
        tasks = torch.stack(tasks)
        entity_mask = torch.stack(entity_mask)
        task_mask = torch.stack(task_mask)
        task_assignments = torch.stack(task_assignments)
        return entities, tasks, entity_mask, task_mask, task_assignments

    def validate(self):
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for entities, tasks, entity_mask, task_mask, targets in self.val_dataloader:
                entities, tasks, entity_mask, task_mask, targets = entities.to(self.device), tasks.to(
                    self.device), entity_mask.to(self.device), task_mask.to(self.device), targets.to(self.device)

                outputs = self.model(entities, tasks, entity_mask, task_mask)
                if outputs is None:
                    continue

                loss = 0
                for i in range(outputs.shape[1]):
                    loss += self.criterion(outputs[:, i, :],
                                           targets[:, i])

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

    def run(self):
        for turn in range(self.epsiode):
            print("current epsiode", turn)
            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(self.num_epochs):
                self.model.train()
                total_loss = 0.0
                layer_losses = {name: 0.0 for name,
                                _ in self.model.named_modules()}

                for entities, tasks, entity_mask, task_mask, task_assignments in self.dataloader:
                    entities = entities.to(self.device)
                    tasks = tasks.to(self.device)
                    entity_mask = entity_mask.to(self.device)
                    task_mask = task_mask.to(self.device)
                    task_assignments = task_assignments.to(self.device)

                    self.optimizer.zero_grad()

                    outputs = self.model(
                        entities, tasks, entity_mask, task_mask)

                    if torch.isnan(outputs).any():
                        print("NaN detected in output")
                        return None
                    if outputs is None:
                        continue

                    # 计算每个平台对应任务的损失
                    loss = 0
                    for i in range(outputs.shape[1]):
                        loss += self.criterion(outputs[:, i, :],
                                               task_assignments[:, i])
                        if torch.isnan(loss).any():
                            print("NaN detected in loss")
                            return None

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0)

                    self.optimizer.step()

                    total_loss += loss.item()

                    # 记录每层的梯度
                    # for name, param in self.model.named_parameters():
                    #     if param.grad is not None:
                    #         self.logger.log_scalar(
                    #             f'{self.name}/gradient_norms/{name}', param.grad.norm().item(), epoch)
                    # if 'combination_layer' in name or 'heads' in name:
                    #     self.logger.log_histogram(
                    #         f'{name}.weight', param.data, epoch)
                    #     self.logger.log_histogram(
                    #         f'{name}.grad', param.grad, epoch)

                avg_train_loss = total_loss / len(self.dataloader)
                avg_val_loss = self.validate()
                self.scheduler.step(avg_val_loss)

                self.logger.log_scalar(
                    self.name+'Loss/train', avg_train_loss, epoch)
                self.logger.log_scalar(
                    self.name+'Loss/val', avg_val_loss, epoch)

                entity_transformer_heads = self.network_config["entity_transformer_heads"]
                task_transformer_heads = self.network_config["task_transformer_heads"]
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{self.num_epochs},Train Loss: {avg_train_loss},Val Loss: {avg_val_loss},entity_num_heads: {entity_transformer_heads},task_transformer_heads: {task_transformer_heads},lr: {current_lr}")

                best_val_loss, patience_counter, stop = self.early_stopping_check(
                    avg_val_loss, best_val_loss, patience_counter, "best_model.pth")
                if stop:
                    break

            self.save_model(self.model_path)

    def save_model(self, path):
        ModelManager.save_model(self.model, self.model_path)

    def load_model(self):
        ModelManager.load_model(self.model, self.model_path, self.device)


if __name__ == "__main__":
    env_config = {
        "max_entities": 20,
        "max_tasks": 20,
        "entity_dim": 6,
        "task_dim": 4,
        "num_samples": 1024,
        "undefined": False
    }

    network_config = {
        "max_entities": env_config["max_entities"],
        "max_tasks": env_config["max_tasks"],
        "entity_input_dim": env_config["entity_dim"],
        "task_input_dim": env_config["task_dim"],
        "entity_transformer_heads": 8,
        "task_transformer_heads": 8,
        "hidden_dim": 64,
        "num_layers": 1,
        "mlp_hidden_dim": 128,
        "entity_headds": env_config["max_entities"],
        "output_dim": env_config["max_tasks"]+1,  # max_tasks增加一个任务编号
        "transfer_dim": 128,
        "use_transformer": False,
        "use_head_mask": False
    }

    training_config = {
        "batch_size": 32,
        "lr": 0.001,
        "num_epochs": 400,
        "patience": 50,
        "epsiode": 100,
        "lr_mode": 'min',
        "factor": 0.95,
        "patience": 100
    }

    trainer = Trainer(env_config, network_config, training_config)
    trainer.run()
