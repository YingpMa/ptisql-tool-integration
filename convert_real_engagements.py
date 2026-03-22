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
        if "privilege" in action_name:
            return "privilege_escalation"
        if "lateral" in action_name or "move" in action_name:
            return "lateral"
        if "exploit" in action_name or "exp" in action_name:
            return "exploit"
        if "scan" in action_name or "enum" in action_name:
            return "discover"
        return action_name

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
            normalized_action_name = normalize_action_name(action_name)
            candidate_action_ids = action_map.get(normalized_action_name, [])

            if not candidate_action_ids:
                print(f"\n[ERROR] Missing action mapping for {normalized_action_name}")
                continue

            selected = None
            for action_id in candidate_action_ids:
                env_copy = gymnasium.make(env_name)
                env_copy.reset()
                try:
                    if len(traj_actions) > 0:
                        replay_obs, _ = env_copy.reset()
                        for replay_action_id in traj_actions:
                            replay_obs, _, replay_terminated, replay_truncated, _ = env_copy.step(replay_action_id)
                            if replay_terminated or replay_truncated:
                                break
                    next_obs_candidate, reward_candidate, terminated_candidate, truncated_candidate, _ = env_copy.step(action_id)
                    if not np.array_equal(obs, next_obs_candidate):
                        selected = (action_id, next_obs_candidate, reward_candidate, terminated_candidate, truncated_candidate)
                        env_copy.close()
                        break
                finally:
                    env_copy.close()

            if selected is None:
                continue

            action_id, next_obs, reward, terminated, truncated = selected
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
