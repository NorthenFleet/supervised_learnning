import torch
from network import DecisionNetworkMultiHead


class InferenceRunner:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        network_config = {
            "model_dim": 128,
            "num_heads": 4,
            "hidden_dim": 512,
            "num_layers": 2,
            "mlp_hidden_dim": 128,
            "output_dim": 5
        }

        model_path = "best_model.pth"


        

        # 初始化模型
        self.model = DecisionNetworkMultiHead(
            env_config["entity_dim"], env_config["task_dim"], network_config["model_dim"],
            network_config["num_heads"], network_config["hidden_dim"], network_config["num_layers"],
            network_config["mlp_hidden_dim"], env_config["max_entities"], network_config["output_dim"]
        )

        # 加载训练好的模型权重
        self.model.load_state_dict(torch.load(model_path))
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

        # 前向推理
        with torch.no_grad():
            output = self.model(entities, tasks, entity_mask, task_mask)

        return output



