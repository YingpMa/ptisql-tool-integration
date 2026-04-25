import copy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPQNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)


def iq_loss(agent, current_Q, current_v, next_v, batch):
    """
    Minimal online IQ-style loss:
    - divergence: kl_fix
    - sampling: value
    """
    gamma = agent.gamma
    obs, next_obs, action, env_reward, done, is_expert = batch

    is_expert = is_expert.bool().squeeze(-1)
    done = done.float()

    y = (1.0 - done) * gamma * next_v
    reward_term = current_Q - y

    expert_reward_term = reward_term[is_expert]

    with torch.no_grad():
        phi_grad = torch.exp(-expert_reward_term)

    softq_loss = -(phi_grad * expert_reward_term).mean()
    value_loss = (current_v - y).mean()
    total_loss = softq_loss + value_loss

    loss_dict = {
        "softq_loss": float(softq_loss.item()),
        "value_loss": float(value_loss.item()),
        "total_loss": float(total_loss.item()),
    }
    return total_loss, loss_dict


class IQAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        alpha: float = 0.1,
        tau: float = 0.005,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.device = torch.device(device)

        self.q_net = MLPQNet(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        # 只是为了兼容原 iq_loss 习惯，实际这里没用复杂 args
        self.args = SimpleNamespace()

    def getV(self, obs, use_target: bool = False):
        net = self.target_net if use_target else self.q_net
        q = net(obs)
        v = self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True)
        return v

    def choose_action(self, obs, sample: bool = True, epsilon: float = 0.1):
        if not isinstance(obs, np.ndarray):
            obs = np.asarray(obs, dtype=np.float32)
        obs = obs.reshape(-1)

        if sample and np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(obs_t)
        return int(q.argmax(dim=1).item())

    def infer_q(self, obs, action):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        act_t = torch.tensor([[int(action)]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            q = self.q_net(obs_t).gather(1, act_t)
        return q.squeeze(0).squeeze(0)

    def infer_v(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            v = self.getV(obs_t, use_target=False)
        return v.squeeze(0).squeeze(0)

    def soft_update_target(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def update(self, expert_batch, policy_batch):
        exp_obs, exp_action, exp_reward, exp_next_obs, exp_done = expert_batch
        pol_obs, pol_action, pol_reward, pol_next_obs, pol_done = policy_batch

        obs = np.concatenate([exp_obs, pol_obs], axis=0)
        action = np.concatenate([exp_action, pol_action], axis=0)
        reward = np.concatenate([exp_reward, pol_reward], axis=0)
        next_obs = np.concatenate([exp_next_obs, pol_next_obs], axis=0)
        done = np.concatenate([exp_done, pol_done], axis=0)

        is_expert = np.concatenate(
            [
                np.ones(len(exp_obs), dtype=np.float32),
                np.zeros(len(pol_obs), dtype=np.float32),
            ],
            axis=0,
        )

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device).view(-1, 1)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).view(-1, 1)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).view(-1, 1)
        is_expert = torch.tensor(is_expert, dtype=torch.float32, device=self.device).view(-1, 1)

        q_all = self.q_net(obs)
        current_Q = q_all.gather(1, action)
        current_v = self.getV(obs, use_target=False)

        with torch.no_grad():
            next_v = self.getV(next_obs, use_target=True)

        batch = (obs, next_obs, action, reward, done, is_expert)
        loss, loss_dict = iq_loss(self, current_Q, current_v, next_v, batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update_target()

        return loss_dict
