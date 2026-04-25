import argparse
import json
import os
import random
from collections import Counter, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, item):
        self.buffer.append(item)

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)

        obs, next_obs, actions, rewards, dones = zip(*batch)

        return (
            torch.tensor(np.array(obs), dtype=torch.float32, device=device),
            torch.tensor(np.array(next_obs), dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
        )

    def size(self):
        return len(self.buffer)


class IQReplayAgent:
    def __init__(self, state_dim, action_dim, args, device):
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.device = device
        self.args = args

        self.log_alpha = torch.tensor(np.log(args.init_temp), device=device)
        self.q_net = QNetwork(state_dim, action_dim, args.hidden_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim, args.hidden_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=args.lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def getV(self, obs):
        q = self.q_net(obs)
        return self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True)

    def get_targetV(self, obs):
        q = self.target_net(obs)
        return self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True)

    def critic(self, obs, action):
        q = self.q_net(obs)
        return q.gather(1, action.long())

    def choose_action(self, state, sample=True):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q = self.q_net(state)
            probs = F.softmax(q / self.alpha, dim=1)

            if sample:
                dist = Categorical(probs)
                action = dist.sample()
            else:
                action = torch.argmax(probs, dim=1)

        return int(action.item())

    def hard_update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


def load_replay(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    trajectories = data["trajectories"] if isinstance(data, dict) else data

    actions = sorted(
        {
            step["action"]
            for traj in trajectories
            for step in traj
        }
    )

    action_to_id = {a: i for i, a in enumerate(actions)}
    id_to_action = {i: a for a, i in action_to_id.items()}

    transitions = []

    for traj in trajectories:
        for step in traj:
            transitions.append(
                (
                    np.array(step["state"], dtype=np.float32),
                    np.array(step["next_state"], dtype=np.float32),
                    action_to_id[step["action"]],
                    float(step.get("reward", 0.0)),
                    float(step.get("done", False)),
                    step["action"],
                )
            )

    return data, trajectories, transitions, action_to_id, id_to_action


def split_transitions(transitions, train_ratio):
    random.shuffle(transitions)
    split = int(len(transitions) * train_ratio)
    return transitions[:split], transitions[split:]


def fill_expert_buffer(buffer, transitions):
    for state, next_state, action_id, reward, done, _ in transitions:
        buffer.add((state, next_state, action_id, reward, done))


def fill_policy_buffer_random(buffer, transitions, capacity):
    pool = list(transitions)

    while buffer.size() < min(capacity, len(pool)):
        state, next_state, action_id, reward, done, _ = random.choice(pool)
        buffer.add((state, next_state, action_id, reward, done))


def concat_samples(policy_batch, expert_batch, device):
    policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch

    obs = torch.cat([policy_obs, expert_obs], dim=0)
    next_obs = torch.cat([policy_next_obs, expert_next_obs], dim=0)
    action = torch.cat([policy_action, expert_action], dim=0)
    reward = torch.cat([policy_reward, expert_reward], dim=0)
    done = torch.cat([policy_done, expert_done], dim=0)

    is_policy = torch.zeros(policy_obs.shape[0], 1, dtype=torch.bool, device=device)
    is_expert = torch.ones(expert_obs.shape[0], 1, dtype=torch.bool, device=device)
    is_expert = torch.cat([is_policy, is_expert], dim=0)

    return obs, next_obs, action, reward, done, is_expert


def iq_loss(agent, current_Q, current_V, next_V, batch):
    args = agent.args
    gamma = agent.gamma

    obs, next_obs, action, env_reward, done, is_expert = batch

    y = (1.0 - done) * gamma * next_V

    reward_iq = current_Q - y
    expert_reward = reward_iq[is_expert]

    loss = -expert_reward.mean()

    if args.loss_type == "value":
        value_loss = (current_V - y).mean()
    elif args.loss_type == "value_expert":
        value_loss = (current_V - y)[is_expert].mean()
    else:
        raise ValueError(f"Unsupported loss_type: {args.loss_type}")

    loss = loss + value_loss

    if args.chi:
        chi2_loss = (expert_reward.pow(2)).mean() / (4.0 * args.chi_alpha)
        loss = loss + chi2_loss
    else:
        chi2_loss = torch.tensor(0.0, device=agent.device)

    return loss, {
        "total_loss": float(loss.item()),
        "softq_loss": float((-expert_reward.mean()).item()),
        "value_loss": float(value_loss.item()),
        "chi2_loss": float(chi2_loss.item()),
        "iq_reward_mean": float(expert_reward.mean().item()),
    }


def iq_update(agent, policy_buffer, expert_buffer, step):
    policy_batch = policy_buffer.sample(agent.batch_size, agent.device)
    expert_batch = expert_buffer.sample(agent.batch_size, agent.device)

    batch = concat_samples(policy_batch, expert_batch, agent.device)
    obs, next_obs, action = batch[0], batch[1], batch[2]

    current_V = agent.getV(obs)

    if agent.args.use_target:
        with torch.no_grad():
            next_V = agent.get_targetV(next_obs)
    else:
        next_V = agent.getV(next_obs)

    current_Q = agent.critic(obs, action)

    loss, loss_dict = iq_loss(agent, current_Q, current_V, next_V, batch)

    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.q_net.parameters(), agent.args.grad_clip)
    agent.optimizer.step()

    if step % agent.args.target_update_interval == 0:
        agent.hard_update_target()

    return loss_dict


@torch.no_grad()
def evaluate(agent, transitions, id_to_action, max_steps):
    agent.q_net.eval()

    total = min(len(transitions), max_steps if max_steps > 0 else len(transitions))

    matched = 0
    reward_sum = 0.0
    pred_counter = Counter()
    expert_counter = Counter()

    for i in range(total):
        state, next_state, expert_action_id, reward, done, expert_action_name = transitions[i]

        pred_action_id = agent.choose_action(state, sample=False)

        pred_counter[id_to_action[pred_action_id]] += 1
        expert_counter[expert_action_name] += 1

        if pred_action_id == expert_action_id:
            matched += 1
            reward_sum += 1.0
        else:
            reward_sum -= 1.0

    matched_ratio = matched / max(total, 1)

    agent.q_net.train()

    return {
        "matched_ratio": matched_ratio,
        "avg_match_reward": reward_sum / max(total, 1),
        "pred_action_distribution": pred_counter,
        "expert_action_distribution": expert_counter,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--replay_path", type=str, default="outputs/replay_iq_650.json")

    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--buffer_capacity", type=int, default=1000000)
    parser.add_argument("--max_steps", type=int, default=0)

    parser.add_argument("--init_temp", type=float, default=1.0)
    parser.add_argument("--target_update_interval", type=int, default=4)
    parser.add_argument("--use_target", action="store_true", default=True)
    parser.add_argument("--loss_type", type=str, default="value", choices=["value", "value_expert"])
    parser.add_argument("--chi", action="store_true", default=True)
    parser.add_argument("--chi_alpha", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=10.0)

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)

    parser.add_argument("--save_dir", type=str, default="outputs/iq_replay")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    data, trajectories, transitions, action_to_id, id_to_action = load_replay(args.replay_path)

    print("=" * 60)
    print(f"[DATA] replay_path = {args.replay_path}")
    print(f"[DATA] trajectories = {len(trajectories)}")
    print(f"[DATA] transitions = {len(transitions)}")
    print(f"[DATA] state_dim = {len(transitions[0][0])}")
    print(f"[DATA] action_dim = {len(action_to_id)}")
    print(f"[DATA] actions = {list(action_to_id.keys())}")
    print(f"[DEVICE] {device}")
    print("=" * 60)

    train_transitions, eval_transitions = split_transitions(transitions, args.train_ratio)

    expert_buffer = ReplayBuffer(args.buffer_capacity // 2)
    policy_buffer = ReplayBuffer(args.buffer_capacity // 2)

    fill_expert_buffer(expert_buffer, train_transitions)
    fill_policy_buffer_random(policy_buffer, train_transitions, args.buffer_capacity // 2)

    print(f"[BUFFER] expert_buffer = {expert_buffer.size()}")
    print(f"[BUFFER] policy_buffer = {policy_buffer.size()}")
    print(f"[EVAL] eval_transitions = {len(eval_transitions)}")
    print("=" * 60)

    state_dim = len(transitions[0][0])
    action_dim = len(action_to_id)

    agent = IQReplayAgent(state_dim, action_dim, args, device)

    os.makedirs(args.save_dir, exist_ok=True)

    recent_losses = deque(maxlen=100)
    recent_iq_rewards = deque(maxlen=100)

    best_matched = -1.0

    for step in range(1, args.episodes + 1):
        losses = iq_update(agent, policy_buffer, expert_buffer, step)

        recent_losses.append(losses["total_loss"])
        recent_iq_rewards.append(losses["iq_reward_mean"])

        if step % args.log_interval == 0:
            print(
                f"[Step {step}] "
                f"loss={np.mean(recent_losses):.6f} "
                f"iq_reward={np.mean(recent_iq_rewards):.6f} "
                f"softq={losses['softq_loss']:.6f} "
                f"value={losses['value_loss']:.6f} "
                f"chi2={losses['chi2_loss']:.6f}"
            )

        if step % args.eval_interval == 0:
            metrics = evaluate(agent, eval_transitions, id_to_action, args.max_steps)

            print("-" * 60)
            print(
                f"[EVAL {step}] "
                f"matched_ratio={metrics['matched_ratio']:.4f} "
                f"avg_match_reward={metrics['avg_match_reward']:.4f}"
            )
            print(f"[EVAL {step}] pred_actions={metrics['pred_action_distribution']}")
            print(f"[EVAL {step}] expert_actions={metrics['expert_action_distribution']}")
            print("-" * 60)

            if metrics["matched_ratio"] > best_matched:
                best_matched = metrics["matched_ratio"]

                save_path = os.path.join(args.save_dir, "best_iq_replay.pt")
                torch.save(
                    {
                        "q_net": agent.q_net.state_dict(),
                        "target_net": agent.target_net.state_dict(),
                        "state_dim": state_dim,
                        "action_dim": action_dim,
                        "hidden_dim": args.hidden_dim,
                        "action_to_id": action_to_id,
                        "id_to_action": id_to_action,
                        "best_matched": best_matched,
                        "args": vars(args),
                    },
                    save_path,
                )

                print(f"[SAVE] best model saved: {save_path}")

    last_path = os.path.join(args.save_dir, "last_iq_replay.pt")
    torch.save(
        {
            "q_net": agent.q_net.state_dict(),
            "target_net": agent.target_net.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": args.hidden_dim,
            "action_to_id": action_to_id,
            "id_to_action": id_to_action,
            "args": vars(args),
        },
        last_path,
    )

    print("=" * 60)
    print("[DONE] Training finished.")
    print(f"[DONE] best_matched = {best_matched:.4f}")
    print(f"[DONE] last model = {last_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
