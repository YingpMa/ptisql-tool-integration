import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import gymnasium
import envs
import numpy as np


def normalize_target(target_str: str):
    if not target_str:
        return None
    target_str = str(target_str).replace(" ", "").strip()
    if target_str.startswith("(") and target_str.endswith(")"):
        parts = target_str[1:-1].split(",")
        if len(parts) == 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                return None
    return None


def parse_action_target(action_text: str):
    marker = "target="
    if marker not in action_text:
        return None
    target_part = action_text.split(marker, 1)[1].split(", cost=", 1)[0].strip()
    return normalize_target(target_part)


def normalize_action_name(action_name: str):
    if not action_name:
        return None

    name = action_name.strip().lower()

    if name == "subnetscan":
        return "subnetscan"
    if name == "osscan":
        return "osscan"
    if name == "servicescan":
        return "servicescan"
    if name == "processscan":
        return "processscan"

    if name in {"http-exp", "httpexp"}:
        return "http-exp"
    if name in {"ssh-exp", "sshexp"}:
        return "ssh-exp"
    if name in {"ftp-exp", "ftpexp"}:
        return "ftp-exp"

    if name in {"tomcat-pe", "tomcatpe"}:
        return "tomcat-pe"
    if name in {"daclsvc-pe", "daclsvcpe", "daclsvc"}:
        return "daclsvc-pe"

    if "subnet" in name:
        return "subnetscan"
    if "os" in name and "scan" in name:
        return "osscan"
    if "service" in name and "scan" in name:
        return "servicescan"
    if "process" in name and "scan" in name:
        return "processscan"
    if "http" in name and ("exp" in name or "exploit" in name):
        return "http-exp"
    if "ssh" in name and ("exp" in name or "exploit" in name):
        return "ssh-exp"
    if "ftp" in name and ("exp" in name or "exploit" in name):
        return "ftp-exp"
    if "tomcat" in name and ("pe" in name or "privilege" in name):
        return "tomcat-pe"
    if "daclsvc" in name and ("pe" in name or "privilege" in name):
        return "daclsvc-pe"

    return None


def build_candidate_index(env_name="nasim:SmallHoneypotPO-v0"):
    env = gymnasium.make(env_name)
    candidates = []

    for action_id in range(env.action_space.n):
        action = env.action_space.get_action(action_id)
        text = str(action)
        target = parse_action_target(text)

        if text.startswith("SubnetScan"):
            key = "subnetscan"
        elif text.startswith("OSScan"):
            key = "osscan"
        elif text.startswith("ServiceScan"):
            key = "servicescan"
        elif text.startswith("ProcessScan"):
            key = "processscan"
        elif text.startswith("Exploit") and "service=http" in text.lower():
            key = "http-exp"
        elif text.startswith("Exploit") and "service=ssh" in text.lower():
            key = "ssh-exp"
        elif text.startswith("Exploit") and "service=ftp" in text.lower():
            key = "ftp-exp"
        elif text.startswith("PrivilegeEscalation") and "process=tomcat" in text.lower():
            key = "tomcat-pe"
        elif text.startswith("PrivilegeEscalation") and "process=daclsvc" in text.lower():
            key = "daclsvc-pe"
        else:
            continue

        candidates.append({
            "action_id": action_id,
            "key": key,
            "target": target,
            "text": text,
        })

    env.close()
    return candidates


def find_action_id(candidates, action_name, target_host):
    norm_name = normalize_action_name(action_name)
    norm_target = normalize_target(target_host)

    if norm_name is None:
        return None

    matched = []
    for item in candidates:
        if item["key"] != norm_name:
            continue
        if norm_target is not None and item["target"] != norm_target:
            continue
        matched.append(item["action_id"])

    if len(matched) == 1:
        return matched[0]

    if len(matched) > 1:
        return matched[0]

    for item in candidates:
        if item["key"] == norm_name:
            return item["action_id"]

    return None


def to_defaultdict_expert():
    return defaultdict(list)


def convert_new_logs_with_env_obs(
    input_root,
    output_pkl,
    env_name="nasim:SmallHoneypotPO-v0",
    limit=None,
):
    envs.register_custom_envs()
    candidates = build_candidate_index(env_name)

    input_root = Path(input_root)
    json_files = sorted(input_root.rglob("engagement_*.json"))
    if limit is not None:
        json_files = json_files[:limit]

    if not json_files:
        raise FileNotFoundError(f"No engagement json files found under: {input_root}")

    expert = to_defaultdict_expert()

    stats = {
        "total_files": 0,
        "loaded_files": 0,
        "skipped_files": 0,
        "total_steps_seen": 0,
        "total_steps_kept": 0,
        "total_action_unmapped": 0,
        "terminated_early": 0,
    }

    action_counter = Counter()
    obs_dim_counter = Counter()

    for json_file in json_files:
        stats["total_files"] += 1

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        steps = data.get("steps", [])
        if not steps:
            print(f"[SKIP] {json_file}: no steps")
            stats["skipped_files"] += 1
            continue

        env = gymnasium.make(env_name)
        obs, _ = env.reset()

        traj_states = []
        traj_next_states = []
        traj_actions = []
        traj_rewards = []
        traj_dones = []

        print(f"\n[PROCESS] {json_file}")

        for idx, step in enumerate(steps, start=1):
            stats["total_steps_seen"] += 1

            action_name = step.get("action")
            target_host = step.get("target_host")
            action_id = find_action_id(candidates, action_name, target_host)

            if action_id is None:
                stats["total_action_unmapped"] += 1
                print(
                    f"[WARN] {json_file.name} step={idx}: "
                    f"unmapped action={action_name}, target={target_host}"
                )
                continue

            current_obs = np.asarray(obs, dtype=np.float32).copy()
            next_obs, reward, terminated, truncated, _ = env.step(action_id)
            next_obs_arr = np.asarray(next_obs, dtype=np.float32).copy()
            done = bool(terminated or truncated)

            traj_states.append(current_obs)
            traj_next_states.append(next_obs_arr)
            traj_actions.append(int(action_id))
            traj_rewards.append(float(reward))
            traj_dones.append(done)

            stats["total_steps_kept"] += 1
            action_counter[normalize_action_name(action_name)] += 1
            obs_dim_counter[current_obs.shape[0]] += 1

            obs = next_obs

            if done:
                stats["terminated_early"] += 1
                print(f"[INFO] {json_file.name}: env terminated early at step {idx}")
                break

        env.close()

        if len(traj_states) == 0:
            print(f"[SKIP] {json_file}: no valid steps kept")
            stats["skipped_files"] += 1
            continue

        # 和原生 expert 文件尽量对齐：每条 trajectory 用 tuple
        expert["states"].append(tuple(traj_states))
        expert["next_states"].append(tuple(traj_next_states))
        expert["actions"].append(tuple(traj_actions))
        expert["rewards"].append(tuple(traj_rewards))
        expert["dones"].append(tuple(traj_dones))
        expert["lengths"].append(len(traj_states))

        stats["loaded_files"] += 1

        print(
            f"[OK] {json_file.name} | kept_steps={len(traj_states)} "
            f"| obs_dim={traj_states[0].shape[0]}"
        )

    output_pkl = Path(output_pkl)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)

    with open(output_pkl, "wb") as f:
        pickle.dump(expert, f)

    print("\n===== SUMMARY =====")
    print("saved_to =", output_pkl)
    print("total_files =", stats["total_files"])
    print("loaded_files =", stats["loaded_files"])
    print("skipped_files =", stats["skipped_files"])
    print("total_steps_seen =", stats["total_steps_seen"])
    print("total_steps_kept =", stats["total_steps_kept"])
    print("total_action_unmapped =", stats["total_action_unmapped"])
    print("terminated_early =", stats["terminated_early"])
    print("num_trajectories =", len(expert["states"]))
    print("total_kept_steps =", sum(expert["lengths"]))
    print("obs_dim_distribution =", dict(obs_dim_counter))
    print("action_distribution =", dict(action_counter))


if __name__ == "__main__":
#    input_root = Path("new_log")
#    output_pkl = Path("experts/new_log_obs.pkl")
#    input_root = Path("new_log_selected")
#    output_pkl = Path("experts/new_log_selected.pkl")
    input_root = Path("new_log_12_only")
    output_pkl = Path("experts/new_log_12_only.pkl")
    convert_new_logs_with_env_obs(
        input_root=input_root,
        output_pkl=output_pkl,
        env_name="nasim:SmallHoneypotPO-v0",
        limit=None,
    )
