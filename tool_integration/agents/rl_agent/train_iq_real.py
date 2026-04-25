import argparse
import random
import numpy as np
import torch

from rl_agent.pt_env import RealPTEnv
from rl_agent.iq_core import IQAgent
from rl_agent.replay_buffer import ReplayBuffer
from rl_agent.expert_dataset import load_expert_transitions


ACTION_NAMES = {
    0: "scan_basic",
    1: "scan_service",
    2: "exploit_bindshell",
    3: "exploit_vsftpd",
    4: "stop",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_expert_batch(expert_transitions, batch_size):
    idxs = np.random.choice(
        len(expert_transitions),
        batch_size,
        replace=len(expert_transitions) < batch_size
    )
    batch = [expert_transitions[i] for i in idxs]
    obs, action, reward, next_obs, done = zip(*batch)
    return (
        np.asarray(obs, dtype=np.float32),
        np.asarray(action, dtype=np.int64),
        np.asarray(reward, dtype=np.float32),
        np.asarray(next_obs, dtype=np.float32),
        np.asarray(done, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_dir", type=str, default="real_logs/rl_env_runs")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--buffer_capacity", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_ip", type=str, default="10.11.202.189")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    expert_transitions, obs_dim = load_expert_transitions(args.expert_dir)
    print(f"Loaded {len(expert_transitions)} expert transitions, obs_dim={obs_dim}")

    env = RealPTEnv(target_ip=args.target_ip)
    action_dim = 5

    agent = IQAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        gamma=args.gamma,
        lr=args.lr,
        alpha=args.alpha,
        tau=args.tau,
        hidden_dim=args.hidden_dim,
        device=device,
    )

    policy_buffer = ReplayBuffer(capacity=args.buffer_capacity)

    # warmup: 先随机采样一些 policy 数据，避免 buffer 为空
    raw_obs = env.reset()
    obs = np.asarray(env.state_to_vector(raw_obs), dtype=np.float)

    for _ in range(args.warmup_steps):
        action = np.random.choice([0, 1])

        print("STEP ACTION:", action)

        raw_next_obs, reward, done, info = env.step(action)
        next_obs = np.asarray(env.state_to_vector(raw_next_obs), dtype=np.float32)

        policy_buffer.push(obs, action, reward, next_obs, done)

        if done:
            raw_obs = env.reset()
            obs = np.asarray(env.state_to_vector(raw_obs), dtype=np.float32)
        else:
            obs = next_obs

    print(f"Warmup done. Policy buffer size: {len(policy_buffer)}")

    # 正式训练
    for episode in range(1, args.episodes + 1):
        raw_obs = env.reset()
        obs = np.asarray(env.state_to_vector(raw_obs), dtype=np.float32)

        episode_reward = 0.0
        action_trace = []
        loss_trace = []

        for step in range(args.max_steps):
            action = agent.choose_action(obs, sample=True, epsilon=args.epsilon)
            if action not in [0, 1]:
                action = np.random.choice([0, 1])
            print("STEP ACTION:", action)
            action_trace.append(ACTION_NAMES.get(action, str(action)))

            raw_next_obs, reward, done, info = env.step(action)
            next_obs = np.asarray(env.state_to_vector(raw_next_obs), dtype=np.float32)

            policy_buffer.push(obs, action, reward, next_obs, done)

            episode_reward += float(reward)
            obs = next_obs

            if len(policy_buffer) >= args.batch_size:
                for _ in range(args.updates_per_step):
                    expert_batch = sample_expert_batch(expert_transitions, args.batch_size)
                    policy_batch = policy_buffer.sample(args.batch_size)
                    loss_dict = agent.update(expert_batch, policy_batch)
                    loss_trace.append(loss_dict["total_loss"])

            if done:
                break

        mean_loss = float(np.mean(loss_trace)) if loss_trace else 0.0

        print(
            f"[Episode {episode:03d}] "
            f"reward={episode_reward:.3f} "
            f"steps={len(action_trace)} "
            f"mean_loss={mean_loss:.6f} "
            f"actions={action_trace}"
        )

    print("Training finished.")


if __name__ == "__main__":
    main()
