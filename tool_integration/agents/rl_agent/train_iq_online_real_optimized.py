import argparse
import json
import os
import random
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

ENV_ACTION_MAP = {
    "scan_basic": 0,
    "scan_service": 1,
    "exploit_bindshell": 2,
    "exploit_vsftpd": 3,
    "exploit_unrealircd": 4,
    "exploit_distccd": 5,
    "exploit_samba": 6,
    "stop": 7,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mean(xs):
    return float(np.mean(xs)) if xs else 0.0


def std(xs):
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def get_valid_action_names(state):
    """
    Agent-level coarse action mask.

    This is intentionally kept the same as the baseline script so that the
    comparison mainly reflects the optimized execution environment:
    caching, staged scanning, and tool-level precondition filtering.
    """
    if not state.get("basic_scanned", False):
        return ["scan_basic"]

    if not state.get("service_scanned", False):
        return ["scan_service"]

    return [
        "exploit_bindshell",
        "exploit_vsftpd",
        "exploit_unrealircd",
        "exploit_distccd",
        "exploit_samba",
        "stop",
    ]


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
        n = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, n)

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


class IQOnlineAgent:
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

    def getV(self, obs, use_target=False):
        net = self.target_net if use_target else self.q_net
        q = net(obs)
        return self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True)

    def critic(self, obs, action):
        q = self.q_net(obs)
        return q.gather(1, action.long())

    def choose_action(self, state_vec, valid_ids=None, sample=True, epsilon=0.1):
        state_vec = np.asarray(state_vec, dtype=np.float32).reshape(-1)

        if valid_ids is None or len(valid_ids) == 0:
            valid_ids = list(range(self.q_net.net[-1].out_features))

        if sample and random.random() < epsilon:
            return random.choice(valid_ids)

        state_t = torch.tensor(
            state_vec,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q = self.q_net(state_t).squeeze(0)

            # Mask invalid actions at agent level.
            masked_q = torch.full_like(q, -1e9)
            masked_q[valid_ids] = q[valid_ids]

            if sample:
                probs = F.softmax(masked_q / self.alpha, dim=0)
                return int(Categorical(probs).sample().item())

            return int(torch.argmax(masked_q).item())

    def hard_update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


def load_expert_replay(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    trajectories = data["trajectories"]
    state_keys = data["state_keys"]

    actions = sorted({step["action"] for traj in trajectories for step in traj})
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
                )
            )

    return data, transitions, state_keys, action_to_id, id_to_action


def state_to_vector(state, state_keys):
    return np.array(
        [float(state.get(k, 0.0)) for k in state_keys],
        dtype=np.float32,
    )


def make_env(args):
    """
    Optimized environment entry point.

    Keep the original baseline environment unchanged.
    Create/copy your optimized environment here:

        tool_integration/agents/rl_agent/pt_env_optimized.py

    That file should expose class RealPTEnv and implement:
    - metrics logging
    - repeated scan caching
    - staged service scanning
    - stronger tool-level precondition filtering
    """
    from tool_integration.agents.rl_agent.pt_env_optimized import RealPTEnv

    return RealPTEnv(
        target_ip=args.target_ip,
        use_metasploit=args.use_metasploit,
    )


def env_step_by_action_name(env, action_name):
    env_action = ENV_ACTION_MAP.get(action_name, 7)

    result = env.step(env_action)

    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        next_state, reward, done, info = result

    return next_state, float(reward), bool(done), info


def fill_expert_buffer(buffer, transitions):
    for transition in transitions:
        buffer.add(transition)


def collect_policy_rollout(
    agent, env, buffer, state_keys, action_to_id, id_to_action, args
):
    state = env.reset()
    episode_reward = 0.0
    actions = []

    for _ in range(args.max_steps):
        state_vec = state_to_vector(state, state_keys)

        valid_actions = get_valid_action_names(state)
        valid_ids = [action_to_id[a] for a in valid_actions if a in action_to_id]

        action_id = agent.choose_action(
            state_vec,
            valid_ids=valid_ids,
            sample=True,
            epsilon=args.epsilon,
        )
        action_name = id_to_action[action_id]

        next_state, reward, done, info = env_step_by_action_name(env, action_name)
        next_state_vec = state_to_vector(next_state, state_keys)

        buffer.add((state_vec, next_state_vec, action_id, reward, float(done)))

        episode_reward += reward
        actions.append(action_name)
        state = next_state

        if done:
            break

    return episode_reward, actions


def concat_samples(policy_batch, expert_batch, device):
    policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = (
        policy_batch
    )
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = (
        expert_batch
    )

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
    obs, next_obs, action, env_reward, done, is_expert = batch

    y = (1.0 - done) * agent.gamma * next_V
    reward_iq = current_Q - y

    expert_reward = reward_iq[is_expert]
    policy_reward = reward_iq[~is_expert]

    softq_loss = -expert_reward.mean()

    if agent.args.loss_type == "value":
        value_loss = (current_V - y).mean()
    else:
        value_loss = (current_V - y)[is_expert].mean()

    loss = softq_loss + value_loss

    # Stabilization: weak penalty for policy states getting too high reward.
    if agent.args.policy_penalty > 0 and policy_reward.numel() > 0:
        loss = loss + agent.args.policy_penalty * policy_reward.mean()

    if agent.args.chi:
        chi2_loss = expert_reward.pow(2).mean() / (4.0 * agent.args.chi_alpha)
        loss = loss + chi2_loss
    else:
        chi2_loss = torch.tensor(0.0, device=agent.device)

    return loss, {
        "total_loss": float(loss.item()),
        "softq_loss": float(softq_loss.item()),
        "value_loss": float(value_loss.item()),
        "chi2_loss": float(chi2_loss.item()),
        "iq_reward_mean": float(expert_reward.mean().item()),
        "policy_iq_reward_mean": (
            float(policy_reward.mean().item()) if policy_reward.numel() else 0.0
        ),
    }


def iq_update(agent, policy_buffer, expert_buffer, step):
    policy_batch = policy_buffer.sample(agent.batch_size, agent.device)
    expert_batch = expert_buffer.sample(agent.batch_size, agent.device)

    batch = concat_samples(policy_batch, expert_batch, agent.device)

    obs, next_obs, action = batch[0], batch[1], batch[2]

    current_V = agent.getV(obs, use_target=False)

    with torch.no_grad():
        next_V = agent.getV(next_obs, use_target=True)

    current_Q = agent.critic(obs, action)

    loss, loss_dict = iq_loss(agent, current_Q, current_V, next_V, batch)

    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.q_net.parameters(), agent.args.grad_clip)
    agent.optimizer.step()

    if step % agent.args.target_update_interval == 0:
        agent.hard_update_target()

    return loss_dict


def fallback_episode_metrics(total_reward, goal, action_counter):
    return {
        "success": int(goal),
        "execution_time": 0.0,
        "total_steps": int(sum(action_counter.values())),
        "tool_calls": 0,
        "invalid_actions": 0,
        "failed_actions": 0,
        "invalid_action_rate": 0.0,
        "failed_action_rate": 0.0,
        "avg_tool_time": 0.0,
        "total_reward": float(total_reward),
    }


@torch.no_grad()
def eval_online(agent, env, state_keys, action_to_id, id_to_action, args):
    rewards = []
    goals = []
    action_counter = Counter()
    episode_metrics = []

    for ep in range(args.eval_eps):
        state = env.reset()
        total_reward = 0.0
        goal = False
        last_info = {}
        ep_actions = Counter()

        for _ in range(args.max_steps):
            state_vec = state_to_vector(state, state_keys)

            valid_actions = get_valid_action_names(state)
            valid_ids = [action_to_id[a] for a in valid_actions if a in action_to_id]

            action_id = agent.choose_action(
                state_vec,
                valid_ids=valid_ids,
                sample=False,
                epsilon=0.0,
            )
            action_name = id_to_action[action_id]

            action_counter[action_name] += 1
            ep_actions[action_name] += 1

            next_state, reward, done, info = env_step_by_action_name(env, action_name)
            last_info = info
            total_reward += reward

            if next_state.get("has_shell", False):
                goal = True

            state = next_state

            if done:
                break

        rewards.append(total_reward)
        goals.append(1.0 if goal else 0.0)

        metrics = last_info.get("metrics")
        if metrics is None:
            metrics = fallback_episode_metrics(total_reward, goal, ep_actions)

        metrics = dict(metrics)
        metrics["total_reward"] = float(total_reward)
        metrics["actions"] = dict(ep_actions)
        episode_metrics.append(metrics)

        print(
            f"[EVAL EP {ep + 1}/{args.eval_eps}] "
            f"success={metrics.get('success', 0)} "
            f"time={metrics.get('execution_time', 0.0):.2f}s "
            f"tool_calls={metrics.get('tool_calls', 0)} "
            f"invalid_rate={metrics.get('invalid_action_rate', 0.0):.3f} "
            f"failed_rate={metrics.get('failed_action_rate', 0.0):.3f}"
        )

    execution_times = [m.get("execution_time", 0.0) for m in episode_metrics]
    tool_calls = [m.get("tool_calls", 0.0) for m in episode_metrics]
    invalid_rates = [m.get("invalid_action_rate", 0.0) for m in episode_metrics]
    failed_rates = [m.get("failed_action_rate", 0.0) for m in episode_metrics]
    avg_tool_times = [m.get("avg_tool_time", 0.0) for m in episode_metrics]

    return {
        "avg_reward": mean(rewards),
        "goal_rate": mean(goals),
        "success_rate": mean([m.get("success", 0) for m in episode_metrics]),
        "avg_execution_time": mean(execution_times),
        "std_execution_time": std(execution_times),
        "avg_tool_calls": mean(tool_calls),
        "std_tool_calls": std(tool_calls),
        "avg_invalid_action_rate": mean(invalid_rates),
        "std_invalid_action_rate": std(invalid_rates),
        "avg_failed_action_rate": mean(failed_rates),
        "std_failed_action_rate": std(failed_rates),
        "avg_tool_time": mean(avg_tool_times),
        "std_tool_time": std(avg_tool_times),
        "actions": action_counter,
        "episodes": episode_metrics,
    }


def save_eval_metrics(metrics, save_dir, step, prefix="optimized_eval"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    serializable = dict(metrics)
    serializable["actions"] = dict(serializable.get("actions", {}))

    path = Path(save_dir) / f"{prefix}_{step}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    return str(path)


def save_model(
    agent,
    path,
    state_dim,
    action_dim,
    hidden_dim,
    action_to_id,
    id_to_action,
    args,
    metrics=None,
):
    torch.save(
        {
            "q_net": agent.q_net.state_dict(),
            "target_net": agent.target_net.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
            "action_to_id": action_to_id,
            "id_to_action": id_to_action,
            "args": vars(args),
            "metrics": metrics or {},
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--replay_path", type=str, default="tool_integration/outputs/replay_iq_650.json"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="tool_integration/outputs/iq_online_real_optimized",
    )

    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--buffer_capacity", type=int, default=100000)

    parser.add_argument("--init_temp", type=float, default=1.0)
    parser.add_argument("--target_update_interval", type=int, default=4)
    parser.add_argument(
        "--loss_type", type=str, default="value", choices=["value", "value_expert"]
    )
    parser.add_argument("--chi", action="store_true", default=True)
    parser.add_argument("--chi_alpha", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--policy_penalty", type=float, default=0.1)

    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--warmup_rollouts", type=int, default=30)
    parser.add_argument("--rollouts_per_update", type=int, default=3)

    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_eps", type=int, default=30)
    parser.add_argument("--log_interval", type=int, default=1)

    parser.add_argument("--target_ip", type=str, default="10.11.202.189")
    parser.add_argument("--use_metasploit", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    data, expert_transitions, state_keys, action_to_id, id_to_action = (
        load_expert_replay(args.replay_path)
    )

    if not expert_transitions:
        raise RuntimeError(f"No expert transitions found in {args.replay_path}")

    state_dim = len(expert_transitions[0][0])
    action_dim = len(action_to_id)

    env = make_env(args)
    eval_env = make_env(args)

    agent = IQOnlineAgent(state_dim, action_dim, args, device)

    expert_buffer = ReplayBuffer(args.buffer_capacity)
    policy_buffer = ReplayBuffer(args.buffer_capacity)

    fill_expert_buffer(expert_buffer, expert_transitions)

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 70)
    print("[TRAIN] IQ online real OPTIMIZED")
    print(f"[TIME] {datetime.now().isoformat(timespec='seconds')}")
    print(f"[DATA] expert_transitions={len(expert_transitions)}")
    print(f"[DATA] state_dim={state_dim}, action_dim={action_dim}")
    print(f"[DATA] action_to_id={action_to_id}")
    print(f"[ENV] target={args.target_ip}, metasploit={args.use_metasploit}")
    print(
        "[ENV] optimized_env=tool_integration.agents.rl_agent.pt_env_optimized.RealPTEnv"
    )
    print(
        "[OPT] expected: metrics + caching + staged scan + tool-level precondition filtering"
    )
    print("=" * 70)

    print("[WARMUP] collecting initial masked policy rollouts")
    for _ in range(args.warmup_rollouts):
        collect_policy_rollout(
            agent,
            env,
            policy_buffer,
            state_keys,
            action_to_id,
            id_to_action,
            args,
        )

    print(f"[BUFFER] expert={expert_buffer.size()} policy={policy_buffer.size()}")

    recent_losses = deque(maxlen=50)
    recent_rewards = deque(maxlen=20)

    best_goal = -1.0
    best_reward = -1e9
    best_efficiency = float("inf")

    final_metrics = None

    for ep in range(1, args.episodes + 1):
        rollout_rewards = []
        rollout_actions = Counter()

        for _ in range(args.rollouts_per_update):
            r, actions = collect_policy_rollout(
                agent,
                env,
                policy_buffer,
                state_keys,
                action_to_id,
                id_to_action,
                args,
            )
            rollout_rewards.append(r)
            rollout_actions.update(actions)

        losses = iq_update(agent, policy_buffer, expert_buffer, ep)

        recent_losses.append(losses["total_loss"])
        recent_rewards.extend(rollout_rewards)

        if ep % args.log_interval == 0:
            print(
                f"[EP {ep}] "
                f"rollout_reward={np.mean(rollout_rewards):.3f} "
                f"recent_reward={np.mean(recent_rewards):.3f} "
                f"loss={np.mean(recent_losses):.6f} "
                f"iq_reward={losses['iq_reward_mean']:.6f} "
                f"policy_iq={losses['policy_iq_reward_mean']:.6f} "
                f"actions={dict(rollout_actions)}"
            )

        if ep % args.eval_interval == 0:
            metrics = eval_online(
                agent,
                eval_env,
                state_keys,
                action_to_id,
                id_to_action,
                args,
            )
            final_metrics = metrics

            eval_path = save_eval_metrics(metrics, args.save_dir, ep)

            print("-" * 70)
            print(
                f"[EVAL {ep}] "
                f"success_rate={metrics['success_rate']:.4f} "
                f"avg_reward={metrics['avg_reward']:.4f} "
                f"avg_time={metrics['avg_execution_time']:.2f}±{metrics['std_execution_time']:.2f}s "
                f"avg_tool_calls={metrics['avg_tool_calls']:.2f}±{metrics['std_tool_calls']:.2f} "
                f"invalid_rate={metrics['avg_invalid_action_rate']:.4f} "
                f"failed_rate={metrics['avg_failed_action_rate']:.4f} "
                f"actions={dict(metrics['actions'])}"
            )
            print(f"[EVAL SAVE] {eval_path}")
            print("-" * 70)

            # Primary selection: higher success rate.
            # Tie-breakers: higher reward, then lower execution time.
            current_goal = metrics["success_rate"]
            current_reward = metrics["avg_reward"]
            current_efficiency = metrics["avg_execution_time"]

            is_better = (
                current_goal > best_goal
                or (current_goal == best_goal and current_reward > best_reward)
                or (
                    current_goal == best_goal
                    and current_reward == best_reward
                    and current_efficiency < best_efficiency
                )
            )

            if is_better:
                best_goal = current_goal
                best_reward = current_reward
                best_efficiency = current_efficiency

                save_path = os.path.join(
                    args.save_dir, "best_iq_online_real_optimized.pt"
                )

                save_model(
                    agent,
                    save_path,
                    state_dim,
                    action_dim,
                    args.hidden_dim,
                    action_to_id,
                    id_to_action,
                    args,
                    metrics,
                )

                print(f"[SAVE] best optimized model saved: {save_path}")

    last_path = os.path.join(args.save_dir, "last_iq_online_real_optimized.pt")

    save_model(
        agent,
        last_path,
        state_dim,
        action_dim,
        args.hidden_dim,
        action_to_id,
        id_to_action,
        args,
        final_metrics,
    )

    print("=" * 70)
    print("[DONE] Optimized training finished.")
    print(f"[DONE] best_success_rate={best_goal:.4f}")
    print(f"[DONE] best_reward={best_reward:.4f}")
    print(f"[DONE] best_avg_execution_time={best_efficiency:.2f}s")
    print(f"[DONE] last_model={last_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
