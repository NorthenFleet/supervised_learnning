import torch
import torch.nn as nn
import torch.optim as optim
from env import Env
from nn import GNN


def train(env, model, num_episodes, lr=0.001, gamma=0.99):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for t in range(100):  # Assuming a maximum of 100 steps per episode
            drone_features, task_features = state

            q_values = model(drone_features, task_features)
            action = torch.argmax(q_values).item()

            next_state, reward, done = env.step(action)

            if done:
                target = reward
            else:
                next_drone_features, next_task_features = next_state
                target = reward + gamma * \
                    torch.max(model(next_drone_features,
                              next_task_features)).item()

            target_f = q_values.clone()
            target_f[action] = target

            loss = criterion(q_values, target_f.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += reward

            if done:
                break

            state = next_state

        print(
            f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")


# 使用示例
if __name__ == '__main__':
    env = Env()
    model = GNN()
