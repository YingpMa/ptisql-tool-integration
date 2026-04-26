import argparse
import json
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn


ALL_ACTIONS = [
    "scan_basic",
    "scan_service",
    "exploit_bindshell",
    "exploit_vsftpd",
    "exploit_unrealircd",
    "exploit_distccd",
    "exploit_samba",
    "stop",
]


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
    id_to_action = {int(k): v for k, v in ckpt["id_to_action"].items()}

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
    return np.array(
        [
            float(state.get(k, 0.0))
            for k in state_keys
        ],
        dtype=np.float32,
    )


def get_valid_action_names(state):
    basic = bool(state.get("basic_scanned", False))
    service = bool(state.get("service_scanned", False))

    if not basic:
        return ["scan_basic"]

    if not service:
        return ["scan_service"]

    return [
        "exploit_bindshell",
        "exploit_vsftpd",
        "exploit_unrealircd",
        "exploit_distccd",
        "exploit_samba",
        "stop",
    ]

def choose_action(
    model,
    state_vec,
    id_to_action,
    device,
    state_dict=None,
    valid_action_mask=False,
):
    x = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(x).squeeze(0).detach().cpu().numpy()

    action_scores = {
        id_to_action[i]: float(logits[i])
        for i in range(len(logits))
        if i in id_to_action
    }

    if valid_action_mask and state_dict is not None:
        valid_actions = get_valid_action_names(state_dict)
        valid_scores = {
            action_name: action_scores.get(action_name, -1e9)
            for action_name in valid_actions
        }
        return max(valid_scores, key=valid_scores.get)

    return max(action_scores, key=action_scores.get)


def choose_random_action(state=None, valid_action_mask=False):
    if valid_action_mask and state is not None:
        return random.choice(get_valid_action_names(state))
    return random.choice(ALL_ACTIONS)


def make_env(args):
    from tool_integration.agents.rl_agent.pt_env import RealPTEnv

    return RealPTEnv(
        target_ip=args.target_ip,
        use_metasploit=args.use_metasploit,
    )


def reset_env(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def step_env(env, action_name):
    action_map = {
        "scan_basic": 0,
        "scan_service": 1,
        "exploit_bindshell": 2,
        "exploit_vsftpd": 3,
        "exploit_unrealircd": 4,
        "exploit_distccd": 5,
        "exploit_samba": 6,
        "stop": 7,
    }

    action_index = action_map.get(action_name, 7)
    result = env.step(action_index)

    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        next_state, reward, done, info = result

    return next_state, float(reward), bool(done), info


def is_goal_reached(env, info, next_state):
    if isinstance(info, dict):
        for key in ["success", "real_success", "goal_reached", "done_success"]:
            if bool(info.get(key, False)):
                return True

    if next_state.get("has_shell", False):
        return True

    if hasattr(env, "state") and env.state.get("has_shell", False):
        return True

    return False


def is_honeypot(reward, info):
    if reward < -90:
        return True

    if isinstance(info, dict):
        for key in ["honeypot", "hit_honeypot", "honeypot_hit"]:
            if key in info:
                return bool(info[key])

    return False


def eval_agent(model, id_to_action, state_keys, device, args):
    env = make_env(args)

    episode_rewards = []
    goals = []
    honeypots = []
    steps_list = []
    action_counter = Counter()
    shell_counter = 0
    metasploit_session_counter = 0

    for ep in range(args.episodes):
        state = reset_env(env)

        episode_reward = 0.0
        goal = False
        honeypot = False
        shell_obtained = False
        metasploit_session = False

        for step in range(args.max_steps):
            if args.model_type == "random":
                action_name = choose_random_action(
                    state=state,
                    valid_action_mask=args.valid_action_mask,
                )
            else:
                state_vec = state_to_vector(state, state_keys)
                action_name = choose_action(
                    model=model,
                    state_vec=state_vec,
                    id_to_action=id_to_action,
                    device=device,
                    state_dict=state,
                    valid_action_mask=args.valid_action_mask,
                )

            action_counter[action_name] += 1

            next_state, reward, done, info = step_env(env, action_name)
            episode_reward += reward

            if is_goal_reached(env, info, next_state):
                goal = True

            if next_state.get("has_shell", False):
                shell_obtained = True

            if (
                isinstance(info, dict)
                and info.get("backend") == "metasploit"
                and info.get("sessions")
            ):
                metasploit_session = True

            if is_honeypot(reward, info):
                honeypot = True

            print(
                f"    [STEP {step + 1}] action={action_name} "
                f"reward={reward:.3f} done={int(done)} "
                f"state={state} info={info}"
            )

            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)
        goals.append(1.0 if goal else 0.0)
        honeypots.append(1.0 if honeypot else 0.0)
        steps_list.append(step + 1)
        shell_counter += int(shell_obtained)
        metasploit_session_counter += int(metasploit_session)

        print(
            f"[EP {ep + 1}] reward={episode_reward:.3f} "
            f"goal={int(goal)} shell={int(shell_obtained)} "
            f"msf_session={int(metasploit_session)} "
            f"honeypot={int(honeypot)} steps={step + 1}"
        )

    return {
        "avg_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "goal_reached_rate": float(np.mean(goals)),
        "honeypot_rate": float(np.mean(honeypots)),
        "avg_steps": float(np.mean(steps_list)),
        "shell_obtained_rate": shell_counter / max(args.episodes, 1),
        "metasploit_session_rate": metasploit_session_counter / max(args.episodes, 1),
        "actions": action_counter,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--model_type", type=str, choices=["iq", "bc", "random"], required=True)
    parser.add_argument("--replay_path", type=str, default="tool_integration/outputs/replay_iq_650.json")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_ip", type=str, default="10.11.202.189")
    parser.add_argument("--use_metasploit", action="store_true")
    parser.add_argument("--valid_action_mask", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    with open(args.replay_path, "r", encoding="utf-8") as f:
        replay = json.load(f)

    state_keys = replay["state_keys"]

    model = None
    id_to_action = None

    if args.model_type != "random":
        if not args.model_path:
            raise ValueError("--model_path is required for iq/bc.")
        model, id_to_action = load_model(args.model_path, args.model_type, device)

    print("=" * 60)
    print(f"[MODEL] {args.model_type}")
    print(f"[PATH] {args.model_path if args.model_path else 'N/A'}")
    print(f"[EPISODES] {args.episodes}")
    print(f"[MAX_STEPS] {args.max_steps}")
    print(f"[TARGET] {args.target_ip}")
    print(f"[USE_METASPLOIT] {args.use_metasploit}")
    print(f"[VALID_ACTION_MASK] {args.valid_action_mask}")
    print(f"[ACTION_MAP] {id_to_action if id_to_action is not None else 'random'}")
    print("=" * 60)

    metrics = eval_agent(
        model=model,
        id_to_action=id_to_action,
        state_keys=state_keys,
        device=device,
        args=args,
    )

    print("=" * 60)
    print(f"[RESULT] model={args.model_type}")
    print(f"avg_reward={metrics['avg_reward']:.4f}")
    print(f"std_reward={metrics['std_reward']:.4f}")
    print(f"goal_reached_rate={metrics['goal_reached_rate']:.4f}")
    print(f"shell_obtained_rate={metrics['shell_obtained_rate']:.4f}")
    print(f"metasploit_session_rate={metrics['metasploit_session_rate']:.4f}")
    print(f"honeypot_rate={metrics['honeypot_rate']:.4f}")
    print(f"avg_steps={metrics['avg_steps']:.4f}")
    print(f"actions={metrics['actions']}")
    print("=" * 60)


if __name__ == "__main__":
    main()