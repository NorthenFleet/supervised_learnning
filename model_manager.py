import torch


class ModelManager:
    @staticmethod
    def save_model(model, path):
        torch.save(model.state_dict(), path)

    @staticmethod
    def load_model(model, path, device):
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
