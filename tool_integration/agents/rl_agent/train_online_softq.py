import json
import random
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from rl_agent.pt_env import RealPTEnv


MODEL_PATH = Path("rl_agent/online_softq_model.pt")
LOG_PATH = Path("rl_agent/online_softq_training_log.json")


class QNet(nn.Module):
    def __init__(self, state_dim=14, action_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s2, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


def select_action(q_net, state_vec, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 4)

    with torch.no_grad():
        x = torch.tensor([state_vec], dtype=torch.float32)
        q_values = q_net(x)[0]
        return int(torch.argmax(q_values).item())


def main():
    random.seed(42)
    torch.manual_seed(42)

    env = RealPTEnv(target_ip="10.11.202.189")

    state_dim = 14
    action_dim = 5
    gamma = 0.95
    lr = 1e-3
    episodes = 60
    batch_size = 32
    min_buffer = 50
    target_update_freq = 20

    epsilon = 1.0
    epsilon_min = 0.10
    epsilon_decay = 0.97

    q_net = QNet(state_dim=state_dim, action_dim=action_dim)
    target_net = QNet(state_dim=state_dim, action_dim=action_dim)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBuffer(capacity=5000)

    training_log = []

    for ep in range(episodes):
        state = env.reset()
        state_vec = env.state_to_vector(state)

        done = False
        ep_reward = 0.0
        actions_taken = []

        while not done:
            action = select_action(q_net, state_vec, epsilon)

            next_state, reward, done, info = env.step(action)
            next_state_vec = env.state_to_vector(next_state)

            replay.add(state_vec, action, reward, next_state_vec, done)

            state_vec = next_state_vec
            ep_reward += reward
            actions_taken.append(env.ACTIONS[action])

            if len(replay) >= min_buffer:
                s, a, r, s2, d = replay.sample(batch_size)

                q_values = q_net(s)
                q_selected = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q = target_net(s2).max(dim=1)[0]
                    target = r + gamma * next_q * (1 - d)

                loss = ((q_selected - target) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if ep % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(
            "episode={0}, reward={1:.2f}, epsilon={2:.3f}, actions={3}".format(
                ep, ep_reward, epsilon, actions_taken
            )
        )

        training_log.append({
            "episode": ep,
            "reward": ep_reward,
            "epsilon": epsilon,
            "actions": actions_taken
        })

    torch.save(
        {
            "model_state_dict": q_net.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
        },
        MODEL_PATH,
    )

    with open(LOG_PATH, "w") as f:
        json.dump(training_log, f, indent=2)

    print("\n[] training finished")
    print("[] model saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
