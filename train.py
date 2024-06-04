import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from env import SampleGenerator, DataPreprocessor
from network import DecisionNetworkMultiHead
from model_manager import ModelManager
from tensorboard_logger import TensorBoardLogger
import os


class Train:
    def __init__(self, env_config, network_config, training_config, data_file=None):
        self.data_preprocessor = DataPreprocessor(
            env_config["max_entities"], env_config["max_tasks"], env_config["entity_dim"], env_config["task_dim"])

        if data_file and os.path.exists(data_file):
            self.dataset = SampleGenerator(
                training_config["num_samples"], self.data_preprocessor, data_file)
        else:
            self.dataset = SampleGenerator(
                training_config["num_samples"], self.data_preprocessor)
            self.dataset.save_data("train_data.h5")

        self.dataloader = DataLoader(
            self.dataset, batch_size=training_config["batch_size"], shuffle=True, collate_fn=self.collate_fn)

        self.val_dataset = SampleGenerator(
            training_config["num_samples"] // 10, self.data_preprocessor, data_file)
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=training_config["batch_size"], shuffle=False, collate_fn=self.collate_fn)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = DecisionNetworkMultiHead(env_config["entity_dim"], env_config["task_dim"], network_config["transfer_dim"],
                                              network_config["entity_num_heads"], network_config["task_num_heads"], network_config["hidden_dim"], network_config["num_layers"], network_config["mlp_hidden_dim"], env_config["max_entities"], network_config["output_dim"] + 1)  # 增加一个任务编号
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=training_config["lr"])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5)

        self.num_epochs = training_config["num_epochs"]
        self.patience = training_config.get("patience", 10)
        self.logger = TensorBoardLogger()

        dummy_entities = torch.zeros(
            (training_config["batch_size"], env_config["max_entities"], env_config["entity_dim"])).to(self.device)
        dummy_tasks = torch.zeros(
            (training_config["batch_size"], env_config["max_tasks"], env_config["task_dim"])).to(self.device)
        dummy_entity_mask = torch.ones(
            (training_config["batch_size"], env_config["max_entities"])).to(self.device)
        dummy_task_mask = torch.ones(
            (training_config["batch_size"], env_config["max_tasks"])).to(self.device)
        
        # self.logger.log_graph(
        #     self.model, (dummy_entities, dummy_tasks, dummy_entity_mask, dummy_task_mask))

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
                if outputs is None:
                    continue

                # Ensure valid_task_assignments and valid_outputs have correct shapes
                valid_mask = task_assignments != -1
                valid_task_assignments = task_assignments[valid_mask]
                valid_outputs = outputs[valid_mask]

                # Check dimensions
                if valid_task_assignments.dim() == 1:
                    valid_task_assignments = valid_task_assignments.unsqueeze(
                        1)
                if valid_outputs.dim() == 2:
                    valid_outputs = valid_outputs.unsqueeze(1)

                assert valid_outputs.shape[:-
                                           1] == valid_task_assignments.shape, "Output and task assignment dimensions do not match"

                loss = 0
                for i in range(valid_outputs.shape[1]):
                    loss += self.criterion(valid_outputs[:, i, :],
                                           valid_task_assignments[:, i])

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.dataloader)
            avg_val_loss = self.validate()
            self.scheduler.step(avg_val_loss)

            self.logger.log_scalar('Loss/train', avg_train_loss, epoch)
            self.logger.log_scalar('Loss/val', avg_val_loss, epoch)

            print(
                f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

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
        "max_entities": 20,
        "max_tasks": 15,
        "entity_dim": 6,
        "task_dim": 4
    }

    network_config = {
        "entity_num_heads": 2,
        "task_num_heads": 2,
        "hidden_dim": 64,
        "num_layers": 2,
        "mlp_hidden_dim": 128,
        "output_dim": 5,  # 增加一个任务编号
        "transfer_dim": 128
    }

    training_config = {
        "num_samples": 1000,
        "batch_size": 32,
        "lr": 0.001,
        "num_epochs": 50,
        "patience": 10
    }

    trainer = Train(env_config, network_config,
                    training_config, data_file="train_data.h5")

    model_path = "best_model.pth"
    try:
        trainer.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(
            f"No existing model found at {model_path}. Starting training from scratch.")

    trainer.train()
    trainer.save_model("best_model.pth")
