import torch
from model_manager import ModelManager
from network import DecisionNetworkMultiHead


class InferenceRunner:
    def __init__(self, env_config):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        network_config = {
            "entity_input_dim": env_config["entity_dim"],
            "task_input_dim": env_config["task_dim"],
            "entity_transformer_heads": 2,
            "task_transformer_heads": 2,
            "hidden_dim": 64,
            "num_layers": 2,
            "mlp_hidden_dim": 128,
            "entity_headds": env_config["max_entities"],
            "output_dim": env_config["max_tasks"]+1,  # max_tasks增加一个任务编号
            "transfer_dim": 128
        }

        model_path = "%s_%s_model.pth" % (
            env_config["max_entities"], env_config["max_tasks"])

        # 初始化模型
        self.model = DecisionNetworkMultiHead(
            env_config["entity_dim"], env_config["task_dim"],
            network_config["transfer_dim"], network_config["entity_transformer_heads"],
            network_config["task_transformer_heads"], network_config["hidden_dim"],
            network_config["num_layers"], network_config["mlp_hidden_dim"],
            env_config["max_entities"], network_config["output_dim"] + 1)  # 增加一个任务编号

        # 加载训练好的模型权重
        self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def run_inference(self, entities, tasks, entity_mask, task_mask):
        # 确保输入数据是单个样本并添加批次维度
        if entities.ndim == 2:
            entities = entities.unsqueeze(0).to(self.device)
        if tasks.ndim == 2:
            tasks = tasks.unsqueeze(0).to(self.device)
        if entity_mask.ndim == 1:
            entity_mask = entity_mask.unsqueeze(0).to(self.device)
        if task_mask.ndim == 1:
            task_mask = task_mask.unsqueeze(0).to(self.device)

        output = self.model.predict(
            entities, tasks, entity_mask, task_mask)

        return output

    def save_model(self, path):
        ModelManager.save_model(self.model, path)

    def load_model(self, path):
        ModelManager.load_model(self.model, path, self.device)
