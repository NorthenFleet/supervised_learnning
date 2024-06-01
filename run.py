import torch
from env import SampleGenerator, DataPreprocessor
from network import DecisionNetworkMultiHead


class InferenceRunner:
    def __init__(self, env_config, network_config, model_path):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # 初始化数据预处理器
        self.data_preprocessor = DataPreprocessor(
            env_config["max_entities"],
            env_config["max_tasks"],
            env_config["entity_dim"],
            env_config["task_dim"]
        )

        # 初始化数据生成器
        self.sample_generator = SampleGenerator(
            num_samples=1, data_preprocessor=self.data_preprocessor)

        # 初始化模型
        self.model = DecisionNetworkMultiHead(
            env_config["entity_dim"], env_config["task_dim"], network_config["model_dim"],
            network_config["num_heads"], network_config["hidden_dim"], network_config["num_layers"],
            network_config["mlp_hidden_dim"], env_config["max_entities"], network_config["output_dim"]
        )

        # 加载训练好的模型权重
        # self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

    def generate_sample(self):
        # 获取一个样本
        sample = self.sample_generator[0]
        padded_entities, padded_tasks, entity_mask, task_mask, targets = sample

        # 转换为适合模型输入的格式（增加一个批次维度）
        padded_entities = padded_entities.unsqueeze(0).to(
            self.device)  # [batch_size, max_entities, entity_dim]
        padded_tasks = padded_tasks.unsqueeze(0).to(
            self.device)        # [batch_size, max_tasks, task_dim]
        entity_mask = entity_mask.unsqueeze(0).to(
            self.device)          # [batch_size, max_entities]
        task_mask = task_mask.unsqueeze(0).to(
            self.device)              # [batch_size, max_tasks]

        return padded_entities, padded_tasks, entity_mask, task_mask, targets

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


def main():
    env_config = {
        "max_entities": 10,
        "max_tasks": 5,
        "entity_dim": 6,
        "task_dim": 4
    }

    network_config = {
        "model_dim": 128,
        "num_heads": 4,
        "hidden_dim": 512,
        "num_layers": 2,
        "mlp_hidden_dim": 128,
        "output_dim": 5
    }

    model_path = "best_model.pth"

    # 创建推理运行器
    runner = InferenceRunner(env_config, network_config, model_path)

    # 生成一个样本
    padded_entities, padded_tasks, entity_mask, task_mask, targets = runner.generate_sample()

    # 运行推理
    output = runner.run_inference(
        padded_entities, padded_tasks, entity_mask, task_mask)
    print(output)


if __name__ == "__main__":
    main()
