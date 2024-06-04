from env import SampleGenerator, DataPreprocessor
from runner import InferenceRunner

class Test():
    def __init__(self, env_config) -> None:
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

    

if __name__ == "__main__":
    env_config = {
        "max_entities": 10,
        "max_tasks": 5,
        "entity_dim": 6,
        "task_dim": 4
        }
     # 创建推理运行器
    runner = InferenceRunner()

    # 生成一个样本
    padded_entities, padded_tasks, entity_mask, task_mask, targets = runner.generate_sample()

    # 运行推理
    output = runner.run_inference(
            padded_entities, padded_tasks, entity_mask, task_mask)
    print(output)
