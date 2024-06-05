import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.multiprocessing import Process, set_start_method, Queue
from env import SampleGenerator, DataPreprocessor
from network import DecisionNetworkMultiHead
from model_manager import ModelManager
from tensorboard_logger import TensorBoardLogger
import os


def train_process(rank, env_config, network_config, training_config, data_file, device, queue):
    data_preprocessor = DataPreprocessor(
        env_config["max_entities"], env_config["max_tasks"], env_config["entity_dim"], env_config["task_dim"])

    if data_file and os.path.exists(data_file):
        dataset = SampleGenerator(
            training_config["num_samples"], data_preprocessor, data_file)
    else:
        dataset = SampleGenerator(
            training_config["num_samples"], data_preprocessor)
        dataset.save_data(data_file)

    dataloader = DataLoader(
        dataset, batch_size=training_config["batch_size"], shuffle=True, collate_fn=collate_fn)

    val_dataset = SampleGenerator(
        training_config["num_samples"] // 10, data_preprocessor, data_file)
    val_dataloader = DataLoader(
        val_dataset, batch_size=training_config["batch_size"], shuffle=False, collate_fn=collate_fn)

    model = DecisionNetworkMultiHead(
        env_config["entity_dim"], env_config["task_dim"], network_config["transfer_dim"],
        network_config["entity_num_heads"], network_config["task_num_heads"],
        network_config["hidden_dim"], network_config["num_layers"],
        network_config["mlp_hidden_dim"], env_config["max_entities"],
        network_config["output_dim"] + 1)  # 增加一个任务编号
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=training_config["lr"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5)

    num_epochs = training_config["num_epochs"]
    patience = training_config.get("patience", 10)
    logger = TensorBoardLogger()

    def collate_fn(batch):
        entities, tasks, entity_mask, task_mask, task_assignments = zip(*batch)
        entities = torch.stack(entities)
        tasks = torch.stack(tasks)
        entity_mask = torch.stack(entity_mask)
        task_mask = torch.stack(task_mask)
        task_assignments = torch.stack(task_assignments)
        return entities, tasks, entity_mask, task_mask, task_assignments

    def validate():
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for entities, tasks, entity_mask, task_mask, targets in val_dataloader:
                entities, tasks, entity_mask, task_mask, targets = entities.to(device), tasks.to(
                    device), entity_mask.to(device), task_mask.to(device), targets.to(device)

                outputs = model(entities, tasks, entity_mask, task_mask)
                if outputs is None:
                    continue

                loss = 0
                for i in range(outputs.shape[1]):
                    loss += criterion(outputs[:, i, :], targets[:, i])

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        return avg_val_loss

    def early_stopping_check(avg_val_loss, best_val_loss, patience_counter, path):
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_model(path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                return best_val_loss, patience_counter, True
        return best_val_loss, patience_counter, False

    def save_model(path):
        ModelManager.save_model(model, path)

    def load_model(path):
        ModelManager.load_model(model, path, device)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for entities, tasks, entity_mask, task_mask, task_assignments in dataloader:
            entities, tasks, entity_mask, task_mask = entities.to(device), tasks.to(
                device), entity_mask.to(device), task_mask.to(device)
            task_assignments = task_assignments.to(device)

            optimizer.zero_grad()

            outputs = model(entities, tasks, entity_mask, task_mask)
            if outputs is None:
                continue

            loss = 0
            for i in range(outputs.shape[1]):
                loss += criterion(outputs[:, i, :], task_assignments[:, i])

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(dataloader)
        avg_val_loss = validate()
        scheduler.step(avg_val_loss)

        logger.log_scalar('Loss/train', avg_train_loss, epoch)
        logger.log_scalar('Loss/val', avg_val_loss, epoch)

        print(
            f"Rank {rank} - Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

        best_val_loss, patience_counter, stop = early_stopping_check(
            avg_val_loss, best_val_loss, patience_counter, f"best_model_rank_{rank}.pth")
        if stop:
            break

    queue.put((rank, best_val_loss))


def main(env_config, network_config, training_config, data_file):
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    num_processes = torch.cuda.device_count()
    processes = []
    queue = Queue()

    for rank in range(num_processes):
        device = torch.device(
            f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        p = Process(target=train_process, args=(rank, env_config,
                    network_config, training_config, data_file, device, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results = []
    while not queue.empty():
        results.append(queue.get())

    print("Training results:", results)


if __name__ == "__main__":
    env_config = {
        "max_entities": 10,
        "max_tasks": 6,
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
        "num_samples": 1024,
        "batch_size": 32,
        "lr": 0.001,
        "num_epochs": 50,
        "patience": 10
    }

    main(env_config, network_config, training_config, data_file="train_data.h5")
