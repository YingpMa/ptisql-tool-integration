import argparse
import json
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
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


def load_model(path, model_type, device):
    ckpt = torch.load(path, map_location=device)

    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]
    hidden_dim = ckpt["hidden_dim"]
    id_to_action = ckpt["id_to_action"]

    # json 里 key 有时会变成 str
    id_to_action = {int(k): v for k, v in id_to_action.items()}

    model = Net(state_dim, action_dim, hidden_dim).to(device)

    if model_type == "bc":
        model.load_state_dict(ckpt["model"])
    elif model_type == "iq":
        model.load_state_dict(ckpt["q_net"])
    else:
        raise ValueError("model_type must be bc or iq")

    model.eval()
    return model, id_to_action


def state_to_vector(state, state_keys):
    vec = []
    for key in state_keys:
        value = state.get(key, 0)
        if isinstance(value, bool):
            vec.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            vec.append(float(value))
        else:
            vec.append(0.0)
    return np.array(vec, dtype=np.float32)


def choose_action(model, state_vec, id_to_action, device, state_dict=None):
    x = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(x).squeeze(0).detach().cpu().numpy()

    allowed = []

    if state_dict is not None:
        basic = bool(state_dict.get("basic_scanned", False))
        service = bool(state_dict.get("service_scanned", False))

        for i, a in id_to_action.items():
            # 1️⃣ 没 basic scan → 只能 basic
            if not basic:
                if a == "scan_basic":
                    allowed.append(i)

            # 2️⃣ basic 有了，但没 service → 只能 service
            elif not service:
                if a == "scan_service":
                    allowed.append(i)

            # 3️⃣ 已经 service_scanned → 禁止再 scan
            else:
                if not a.startswith("scan"):
                    allowed.append(i)

    if not allowed:
        allowed = list(id_to_action.keys())

    best_i = max(allowed, key=lambda i: logits[i])
    return id_to_action[int(best_i)]



def make_env():
    from rl_agent.pt_env import RealPTEnv
    return RealPTEnv()
    from rl_agent.pt_env import RealPTEnv
    return RealPTEnv()
    """
    这里按你的项目结构自动尝试导入 RealPTEnv。
    如果报错，把你真实 env 文件名发我，我帮你改 import。
    """
    try:
        from rl_agent.real_pt_env import RealPTEnv
        return RealPTEnv()
    except Exception:
        pass

    try:
        from rl_agent.env import RealPTEnv
        return RealPTEnv()
    except Exception:
        pass

    try:
        from envs.real_pt_env import RealPTEnv
        return RealPTEnv()
    except Exception:
        pass

    raise ImportError(
        "Cannot import RealPTEnv. Please tell me where your environment class is."
    )


def reset_env(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def step_env(env, action_name):
    ACTION_MAP = {
        "scan_basic": 0,
        "scan_service": 1,
        "exploit_bindshell": 2,
        "exploit_vsftpd": 3,
        "stop": 4,
        "exploit_samba": 2,
        "exploit_unrealircd": 2,
        "exploit_distccd": 2,
    }

    # enforce scan-before-exploit
    if action_name.startswith("exploit") and hasattr(env, "state") and not env.state.get("service_scanned", False):
        action_name = "scan_service"

    action_index = ACTION_MAP.get(action_name, 4)
    result = env.step(action_index)

    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        next_state, reward, done, info = result

    return next_state, float(reward), bool(done), info


def is_goal_reached(env, info):
    if isinstance(info, dict):
        for key in ["goal_reached", "success", "done_success"]:
            if key in info:
                return bool(info[key])

    if hasattr(env, "goal_reached"):
        try:
            return bool(env.goal_reached())
        except Exception:
            return False

    return False


def is_honeypot(reward, info):
    if reward < -90:
        return True

    if isinstance(info, dict):
        for key in ["honeypot", "hit_honeypot", "honeypot_hit"]:
            if key in info:
                return bool(info[key])

    return False


def eval_agent(model, id_to_action, state_keys, device, episodes, max_steps):
    env = make_env()

    rewards = []
    goals = []
    honeypots = []
    steps_list = []
    action_counter = Counter()

    for ep in range(episodes):
        state = reset_env(env)

        total_reward = 0.0
        goal = False
        honeypot = False

        for step in range(max_steps):
            state_vec = state_to_vector(state, state_keys)
            action_name = choose_action(model, state_vec, id_to_action, device, state)

            action_counter[action_name] += 1

            next_state, reward, done, info = step_env(env, action_name)

            print(
                f"    [STEP {step + 1}] action={action_name} "
                f"reward={reward:.3f} done={int(done)} "
                f"state={state} info={info}"
            )

            total_reward += reward

            if is_honeypot(reward, info):
                honeypot = True

            if is_goal_reached(env, info):
                goal = True

            state = next_state

            if done:
                break

        rewards.append(total_reward)
        goals.append(1.0 if goal else 0.0)
        honeypots.append(1.0 if honeypot else 0.0)
        steps_list.append(step + 1)

        print(
            f"[EP {ep + 1}] reward={total_reward:.3f} "
            f"goal={int(goal)} honeypot={int(honeypot)} steps={step + 1}"
        )

    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "goal_reached_rate": float(np.mean(goals)),
        "honeypot_rate": float(np.mean(honeypots)),
        "avg_steps": float(np.mean(steps_list)),
        "actions": action_counter,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["iq", "bc"], required=True)
    parser.add_argument("--replay_path", type=str, default="outputs/replay_iq_650.json")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    with open(args.replay_path, "r", encoding="utf-8") as f:
        replay = json.load(f)

    state_keys = replay["state_keys"]

    model, id_to_action = load_model(args.model_path, args.model_type, device)

    print("=" * 60)
    print(f"[MODEL] {args.model_type}")
    print(f"[PATH] {args.model_path}")
    print(f"[EPISODES] {args.episodes}")
    print(f"[MAX_STEPS] {args.max_steps}")
    print("=" * 60)

    metrics = eval_agent(
        model=model,
        id_to_action=id_to_action,
        state_keys=state_keys,
        device=device,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )

    print("=" * 60)
    print(f"[RESULT] model={args.model_type}")
    print(f"avg_reward={metrics['avg_reward']:.4f}")
    print(f"std_reward={metrics['std_reward']:.4f}")
    print(f"goal_reached_rate={metrics['goal_reached_rate']:.4f}")
    print(f"honeypot_rate={metrics['honeypot_rate']:.4f}")
    print(f"avg_steps={metrics['avg_steps']:.4f}")
    print(f"actions={metrics['actions']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
