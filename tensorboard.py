from torch.utils.tensorboard import SummaryWriter
import torch


class TensorBoardLogger:
    def __init__(self, log_dir="runs"):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_graph(self, model, dummy_inputs):
        self.writer.add_graph(model, dummy_inputs)

    def close(self):
        self.writer.close()
