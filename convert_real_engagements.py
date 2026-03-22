import json
import pickle
from pathlib import Path

import gymnasium
import envs
import numpy as np


def build_action_map(env_name="nasim:SmallHoneypotPO-v0"):
    env = gymnasium.make(env_name)

    def normalize_action_name(action_name: str):
        action_name = action_name.lower()
        if "privilege" in action_name:
            return "privilege_escalation"
        if "lateral" in action_name or "move" in action_name:
            return "lateral"
        if "exploit" in action_name or "exp" in action_name:
            return "exploit"
        if "scan" in action_name or "enum" in action_name:
            return "discover"
        return action_name

    action_map = {}

    for action_id in range(env.action_space.n):
        print(action_id, str(env.action_space.get_action(action_id)))

    for i in range(env.action_space.n):
        a = env.action_space.get_action(i)
        text = str(a)

        if text.startswith("SubnetScan"):
            action_name = "SubnetScan"
        elif text.startswith("OSScan"):
            action_name = "OSScan"
        elif text.startswith("ServiceScan"):
            action_name = "ServiceScan"
        elif text.startswith("ProcessScan"):
            action_name = "ProcessScan"
        elif text.startswith("PrivilegeEscalation"):
            if "process=tomcat" in text:
                action_name = "Tomcat-PE"
            elif "process=daclsvc" in text:
                action_name = "Daclsvc-PE"
            else:
                action_name = "PrivilegeEscalation"
        elif text.startswith("Exploit"):
            if "service=ftp" in text:
                action_name = "FTP-EXP"
            elif "service=http" in text:
                action_name = "HTTP-EXP"
            elif "service=ssh" in text:
                action_name = "SSH-EXP"
            else:
                raise ValueError(f"Unknown exploit type in action: {text}")
        else:
            raise ValueError(f"Unknown action type: {text}")

        normalized_action_name = normalize_action_name(action_name)
        action_map.setdefault(normalized_action_name, []).append(i)

    env.close()
    return action_map


def convert_engagement_jsons(
    json_files,
    output_pkl,
    env_name="nasim:SmallHoneypotPO-v0",
):
    envs.register_custom_envs()
    action_map = build_action_map(env_name)

    def normalize_action_name(action_name: str):
        action_name = action_name.lower()
        if "servicescan" in action_name:
            return "servicescan"
        if "osscan" in action_name:
            return "osscan"
        if "subnetscan" in action_name or "scan" in action_name or "enum" in action_name:
            return "subnetscan"
        if "processscan" in action_name:
            return "processscan"
        if "http-exp" in action_name:
            return "http-exp"
        if "ssh-exp" in action_name:
            return "ssh-exp"
        if "ftp-exp" in action_name:
            return "ftp-exp"
        if "tomcat-pe" in action_name:
            return "tomcat-pe"
        if "daclsvc" in action_name:
            return "daclsvc"
        return action_name

    def select_best_action(env, action_name, target, obs):
        normalized_action_name = normalize_action_name(action_name)
        candidate_action_ids = []

        for aid in range(env.action_space.n):
            action = env.action_space.get_action(aid)
            text = str(action).lower()

            if normalized_action_name == "servicescan" and "servicescan" in text:
                candidate_action_ids.append(aid)
            elif normalized_action_name == "osscan" and "osscan" in text:
                candidate_action_ids.append(aid)
            elif normalized_action_name == "subnetscan" and "subnetscan" in text:
                candidate_action_ids.append(aid)
            elif normalized_action_name == "processscan" and "processscan" in text:
                candidate_action_ids.append(aid)
            elif normalized_action_name == "http-exp" and "service=http" in text:
                candidate_action_ids.append(aid)
            elif normalized_action_name == "ssh-exp" and "service=ssh" in text:
                candidate_action_ids.append(aid)
            elif normalized_action_name == "ftp-exp" and "service=ftp" in text:
                candidate_action_ids.append(aid)
            elif normalized_action_name == "tomcat-pe" and "process=tomcat" in text:
                candidate_action_ids.append(aid)
            elif normalized_action_name == "daclsvc" and "process=daclsvc" in text:
                candidate_action_ids.append(aid)

        for action_id in candidate_action_ids:
            next_obs, _, _, _, _ = env.step(action_id)
            if not np.array_equal(obs, next_obs):
                return action_id

        return None

    expert = {
        "states": [],
        "actions": [],
        "rewards": [],
        "next_states": [],
        "dones": [],
        "lengths": [],
    }

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 每条真实 engagement 单独 reset 一次环境
        env = gymnasium.make(env_name)
        obs, _ = env.reset()

        traj_states = []
        traj_actions = []
        traj_rewards = []
        traj_next_states = []
        traj_dones = []

        print(f"\nProcessing {json_file} ...")

        for idx, step in enumerate(data["steps"]):
            action_name = step["action"]
            target = step["target_host"].replace(" ", "")
            print("JSON_ACTION:", action_name, target)
            action_id = select_best_action(env, action_name, target, obs)

            if action_id is None:
                continue

            next_obs, reward, terminated, truncated, _ = env.step(action_id)
            done = bool(terminated or truncated)

            traj_states.append(np.array(obs, dtype=np.float32))
            traj_actions.append(int(action_id))
            traj_rewards.append(float(reward))
            traj_next_states.append(np.array(next_obs, dtype=np.float32))
            traj_dones.append(done)

            obs = next_obs

            if done:
                print(
                    f"[INFO] {json_file.name}: environment terminated early "
                    f"at step {idx + 1}"
                )
                break

        expert["states"].append(traj_states)
        expert["actions"].append(traj_actions)
        expert["rewards"].append(traj_rewards)
        expert["next_states"].append(traj_next_states)
        expert["dones"].append(traj_dones)
        expert["lengths"].append(len(traj_states))

        if len(traj_states) > 0:
            print(
                f"Loaded {json_file} | recorded_steps={len(traj_states)} "
                f"| state_dim={traj_states[0].shape[0]}"
            )
        else:
            print(f"Loaded {json_file} | recorded_steps=0")

        env.close()

    with open(output_pkl, "wb") as f:
        pickle.dump(expert, f)

    print("\nSaved expert dataset to:", output_pkl)
    print("Num trajectories:", len(expert["states"]))
    print("Lengths:", expert["lengths"])
    print("Total steps:", sum(expert["lengths"]))


if __name__ == "__main__":
    input_dir = Path("real_engagements")
    json_files = sorted(input_dir.glob("engagement_*.json"))
    output_pkl = "experts/real_engagements.pkl"

    convert_engagement_jsons(json_files, output_pkl)
