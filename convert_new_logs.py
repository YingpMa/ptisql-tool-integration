import json
import pickle
from collections import Counter
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


def semantic_match(normalized_name: str, action_text: str):
    text = action_text.lower()

    if normalized_name == "subnetscan":
        return text.startswith("subnetscan")
    if normalized_name == "osscan":
        return text.startswith("osscan")
    if normalized_name == "servicescan":
        return text.startswith("servicescan")
    if normalized_name == "processscan":
        return text.startswith("processscan")
    if normalized_name == "http-exp":
        return text.startswith("exploit") and "service=http" in text
    if normalized_name == "ssh-exp":
        return text.startswith("exploit") and "service=ssh" in text
    if normalized_name == "ftp-exp":
        return text.startswith("exploit") and "service=ftp" in text
    if normalized_name == "tomcat-pe":
        return text.startswith("privilegeescalation") and "process=tomcat" in text
    if normalized_name == "daclsvc-pe":
        return text.startswith("privilegeescalation") and "process=daclsvc" in text

    return False


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


def find_action_id(env, candidates, action_name, target_host):
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
        # 保守处理：取第一个
        return matched[0]

    # fallback：忽略 target，只按语义找
    for item in candidates:
        if item["key"] == norm_name:
            return item["action_id"]

    return None


def convert_engagement_jsons(
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

    expert = {
        "states": [],
        "actions": [],
        "rewards": [],
        "next_states": [],
        "dones": [],
        "lengths": [],
        "files": [],
    }

    stats = {
        "total_files": 0,
        "loaded_files": 0,
        "skipped_files": 0,
        "total_steps_seen": 0,
        "total_steps_kept": 0,
        "total_action_unmapped": 0,
        "total_bad_state": 0,
    }

    action_counter = Counter()

    for json_file in json_files:
        stats["total_files"] += 1

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        steps = data.get("steps", [])
        if not steps:
            print(f"[SKIP] {json_file}: no steps")
            stats["skipped_files"] += 1
            continue

        traj_states = []
        traj_actions = []
        traj_rewards = []
        traj_next_states = []
        traj_dones = []

        env = gymnasium.make(env_name)

        print(f"\n[PROCESS] {json_file}")

        for idx, step in enumerate(steps, start=1):
            stats["total_steps_seen"] += 1

            state = step.get("state")
            next_state = step.get("next_state")
            action_name = step.get("action")
            target_host = step.get("target_host")
            reward = step.get("reward", 0.0)
            done = bool(step.get("done", False))

            if state is None or next_state is None:
                stats["total_bad_state"] += 1
                print(f"[WARN] {json_file.name} step={idx}: missing state/next_state")
                continue

            try:
                state_arr = np.asarray(state, dtype=np.float32)
                next_state_arr = np.asarray(next_state, dtype=np.float32)
            except Exception:
                stats["total_bad_state"] += 1
                print(f"[WARN] {json_file.name} step={idx}: bad state format")
                continue

            if state_arr.ndim != 1 or next_state_arr.ndim != 1:
                stats["total_bad_state"] += 1
                print(f"[WARN] {json_file.name} step={idx}: non-1D state")
                continue

            action_id = find_action_id(env, candidates, action_name, target_host)
            if action_id is None:
                stats["total_action_unmapped"] += 1
                print(
                    f"[WARN] {json_file.name} step={idx}: "
                    f"unmapped action={action_name}, target={target_host}"
                )
                continue

            traj_states.append(state_arr)
            traj_actions.append(int(action_id))
            traj_rewards.append(float(reward))
            traj_next_states.append(next_state_arr)
            traj_dones.append(done)

            stats["total_steps_kept"] += 1
            action_counter[normalize_action_name(action_name)] += 1

        env.close()

        if len(traj_states) == 0:
            print(f"[SKIP] {json_file}: no valid steps kept")
            stats["skipped_files"] += 1
            continue

        expert["states"].append(traj_states)
        expert["actions"].append(traj_actions)
        expert["rewards"].append(traj_rewards)
        expert["next_states"].append(traj_next_states)
        expert["dones"].append(traj_dones)
        expert["lengths"].append(len(traj_states))
        expert["files"].append(str(json_file))

        stats["loaded_files"] += 1

        print(
            f"[OK] {json_file.name} | kept_steps={len(traj_states)} "
            f"| state_dim={traj_states[0].shape[0]}"
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
    print("total_bad_state =", stats["total_bad_state"])
    print("num_trajectories =", len(expert["states"]))
    print("total_kept_steps =", sum(expert["lengths"]))
    print("action_distribution =", dict(action_counter))


if __name__ == "__main__":
    # 先跑 new_log
    input_root = Path("new_log")
    output_pkl = Path("experts/new_log.pkl")

    # 调试时可改成 limit=10
    convert_engagement_jsons(
        input_root=input_root,
        output_pkl=output_pkl,
        env_name="nasim:SmallHoneypotPO-v0",
        limit=None,
    )