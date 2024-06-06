from env import SampleGenerator, DataPreprocessor
from runner import InferenceRunner


class Data_Sample():
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
            env_config["max_entities"], env_config, self.data_preprocessor)

    def generate_sample(self):
        # 获取一个样本
        sample = self.sample_generator[0]
        padded_entities, padded_tasks, entity_mask, task_mask, targets = sample

        return padded_entities, padded_tasks, entity_mask, task_mask, targets


if __name__ == "__main__":
    env_config = {
        "max_entities": 1,
        "max_tasks": 5,
        "entity_dim": 6,
        "task_dim": 4,
        "undefined": False
    }
    data_sample = Data_Sample(env_config)

    # 创建推理运行器
    runner = InferenceRunner(env_config)

    # 生成一个样本
    padded_entities, padded_tasks, entity_mask, task_mask, targets = data_sample.generate_sample()
    print("entities:", padded_entities)
    print("tasks:", padded_tasks)

    # 运行推理
    network_output = runner.run_inference(
        padded_entities, padded_tasks, entity_mask, task_mask)

    target = data_sample.sample_generator.get_target(
        padded_entities, padded_tasks)

    print("output:", network_output[0])
    print("target:", target)
