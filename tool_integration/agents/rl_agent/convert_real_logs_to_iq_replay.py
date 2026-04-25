import argparse
import json
import os
from glob import glob
from collections import Counter


STATE_KEYS = [
    "num_open_ports",
    "has_ftp",
    "has_ssh",
    "has_telnet",
    "has_http",
    "has_mysql",
    "has_postgresql",
    "has_bindshell_1524",
    "has_samba",
    "has_tomcat",
    "has_vsftpd_234",
    "has_unrealircd",
    "has_distccd",
    "has_shell",
    "basic_scanned",
    "service_scanned",
    "failed_attempts",
    "successful_exploits",
]


def state_to_vector(state):
    vec = []
    for key in STATE_KEYS:
        value = state.get(key, 0)

        if isinstance(value, bool):
            vec.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            vec.append(float(value))
        else:
            vec.append(0.0)

    return vec


def load_one_run(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history", [])
    if not history:
        return None

    trajectory = []

    for i, step in enumerate(history):
        state = state_to_vector(step["state"])
        action = step["action"]
        reward = float(step.get("reward", 0.0))

        if i + 1 < len(history):
            next_state = state_to_vector(history[i + 1]["state"])
            done = bool(step.get("done", False))
        else:
            next_state = state
            done = True

        trajectory.append(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
            }
        )

    return trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="real_logs/rl_env_runs")
    parser.add_argument("--output_path", type=str, default="outputs/replay_iq_650.json")
    args = parser.parse_args()

    files = sorted(glob(os.path.join(args.input_dir, "*.json")))

    trajectories = []
    action_counter = Counter()
    total_steps = 0
    skipped = 0

    for path in files:
        traj = load_one_run(path)

        if traj is None or len(traj) == 0:
            skipped += 1
            continue

        trajectories.append(traj)
        total_steps += len(traj)

        for step in traj:
            action_counter[step["action"]] += 1

    output = {
        "state_keys": STATE_KEYS,
        "num_trajectories": len(trajectories),
        "total_steps": total_steps,
        "action_distribution": dict(action_counter),
        "trajectories": trajectories,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("=" * 60)
    print(f"[OK] Input dir: {args.input_dir}")
    print(f"[OK] Files found: {len(files)}")
    print(f"[OK] Trajectories saved: {len(trajectories)}")
    print(f"[OK] Skipped: {skipped}")
    print(f"[OK] Total steps: {total_steps}")
    print(f"[OK] Output path: {args.output_path}")
    print(f"[OK] Action distribution: {action_counter}")
    print("=" * 60)


if __name__ == "__main__":
    main()
