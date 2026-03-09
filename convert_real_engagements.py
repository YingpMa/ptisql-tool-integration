import json
import pickle
from pathlib import Path

import gymnasium
import envs
import numpy as np


def build_action_map(env_name="nasim:SmallHoneypotPO-v0"):
    envs.register_custom_envs()
    env = gymnasium.make(env_name, render_mode="human")

    action_map = {}

    for i in range(env.action_space.n):
        a = env.action_space.get_action(i)
        text = str(a)

        target_part = text.split("target=")[1].split(", cost=")[0]
        target = target_part.replace(" ", "")

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

        action_map[(action_name, target)] = i

    return action_map


def convert_engagement_jsons(json_files, output_pkl):

    action_map = build_action_map()

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

        traj_states = []
        traj_actions = []
        traj_rewards = []
        traj_next_states = []
        traj_dones = []

        for step in data["steps"]:

            action_name = step["action"]
            target = step["target_host"].replace(" ", "")

            key = (action_name, target)

            if key not in action_map:
                print(f"\n[ERROR] Missing action mapping for {key}")
                print("Available actions for this target:")

                for k, v in sorted(action_map.items()):
                    if k[1] == target:
                        print("   ", k, "->", v)

                raise KeyError(f"Action mapping not found for {key}")

            action_id = action_map[key]

            traj_states.append(np.array(step["state"], dtype=np.float32))
            traj_actions.append(int(action_id))
            traj_rewards.append(float(step["reward"]))
            traj_next_states.append(np.array(step["next_state"], dtype=np.float32))
            traj_dones.append(bool(step["done"]))

        expert["states"].append(traj_states)
        expert["actions"].append(traj_actions)
        expert["rewards"].append(traj_rewards)
        expert["next_states"].append(traj_next_states)
        expert["dones"].append(traj_dones)
        expert["lengths"].append(len(traj_states))

        print(f"Loaded {json_file} | steps={len(traj_states)}")

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
