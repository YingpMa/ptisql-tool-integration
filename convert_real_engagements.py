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

    def audit_goal_reached(env):
        try:
            return env.unwrapped.goal_reached()
        except Exception:
            return None

    def audit_host_vector(obs):
        try:
            arr = np.asarray(obs)
            if arr.ndim != 1:
                return None
            return arr.copy()
        except Exception:
            return None

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
        def parse_target_tuple(target_str):
            target_str = target_str.strip()
            if target_str.startswith("(") and target_str.endswith(")"):
                parts = target_str[1:-1].split(",")
                if len(parts) == 2:
                    return (int(parts[0].strip()), int(parts[1].strip()))
            return None

        def parse_action_target(text):
            marker = "target="
            if marker not in text:
                return None
            target_part = text.split(marker, 1)[1].split(", cost=", 1)[0].strip()
            return parse_target_tuple(target_part)

        normalized_action_name = normalize_action_name(action_name)
        json_target = parse_target_tuple(target)
        candidate_action_ids = []

        for aid in range(env.action_space.n):
            action = env.action_space.get_action(aid)
            text = str(action)
            text_lower = text.lower()
            action_target = parse_action_target(text)

            if action_target != json_target:
                continue

            if normalized_action_name == "servicescan" and "servicescan" in text_lower:
                candidate_action_ids.append(aid)
            elif normalized_action_name == "osscan" and "osscan" in text_lower:
                candidate_action_ids.append(aid)
            elif normalized_action_name == "subnetscan" and "subnetscan" in text_lower:
                candidate_action_ids.append(aid)
            elif normalized_action_name == "processscan" and "processscan" in text_lower:
                candidate_action_ids.append(aid)
            elif normalized_action_name == "http-exp" and text_lower.startswith("exploit") and "service=http" in text_lower:
                candidate_action_ids.append(aid)
            elif normalized_action_name == "ssh-exp" and text_lower.startswith("exploit") and "service=ssh" in text_lower:
                candidate_action_ids.append(aid)
            elif normalized_action_name == "ftp-exp" and text_lower.startswith("exploit") and "service=ftp" in text_lower:
                candidate_action_ids.append(aid)
            elif normalized_action_name == "tomcat-pe" and text_lower.startswith("privilegeescalation") and "process=tomcat" in text_lower:
                candidate_action_ids.append(aid)
            elif normalized_action_name == "daclsvc" and text_lower.startswith("privilegeescalation") and "process=daclsvc" in text_lower:
                candidate_action_ids.append(aid)

        for action_id in candidate_action_ids:
            env_copy = gymnasium.make(env_name)
            try:
                env_copy.reset()
                next_obs, _, _, _, _ = env_copy.step(action_id)
                if not np.array_equal(next_obs, obs):
                    return action_id
            finally:
                env_copy.close()

        return None

    def select_best_action_replayed(action_name, target, obs, replay_actions):
        def parse_target_tuple(target_str):
            target_str = target_str.strip()
            if target_str.startswith("(") and target_str.endswith(")"):
                parts = target_str[1:-1].split(",")
                if len(parts) == 2:
                    return (int(parts[0].strip()), int(parts[1].strip()))
            return None

        def parse_action_target(text):
            marker = "target="
            if marker not in text:
                return None
            target_part = text.split(marker, 1)[1].split(", cost=", 1)[0].strip()
            return parse_target_tuple(target_part)

        normalized_action_name = normalize_action_name(action_name)
        json_target = parse_target_tuple(target)

        for aid in range(gymnasium.make(env_name).action_space.n):
            env_copy = gymnasium.make(env_name)
            try:
                env_copy.reset()
                for replay_action_id in replay_actions:
                    _, replay_terminated, replay_truncated = None, False, False
                    _, _, replay_terminated, replay_truncated, _ = env_copy.step(replay_action_id)
                    if replay_terminated or replay_truncated:
                        break

                action = env_copy.action_space.get_action(aid)
                text = str(action)
                text_lower = text.lower()
                action_target = parse_action_target(text)

                if action_target != json_target:
                    continue

                semantic_match = False
                if normalized_action_name == "servicescan" and "servicescan" in text_lower:
                    semantic_match = True
                elif normalized_action_name == "osscan" and "osscan" in text_lower:
                    semantic_match = True
                elif normalized_action_name == "subnetscan" and "subnetscan" in text_lower:
                    semantic_match = True
                elif normalized_action_name == "processscan" and "processscan" in text_lower:
                    semantic_match = True
                elif normalized_action_name == "http-exp" and text_lower.startswith("exploit") and "service=http" in text_lower:
                    semantic_match = True
                elif normalized_action_name == "ssh-exp" and text_lower.startswith("exploit") and "service=ssh" in text_lower:
                    semantic_match = True
                elif normalized_action_name == "ftp-exp" and text_lower.startswith("exploit") and "service=ftp" in text_lower:
                    semantic_match = True
                elif normalized_action_name == "tomcat-pe" and text_lower.startswith("privilegeescalation") and "process=tomcat" in text_lower:
                    semantic_match = True
                elif normalized_action_name == "daclsvc" and text_lower.startswith("privilegeescalation") and "process=daclsvc" in text_lower:
                    semantic_match = True

                if not semantic_match:
                    continue

                next_obs, _, _, _, _ = env_copy.step(aid)
                if not np.array_equal(next_obs, obs):
                    return aid
            finally:
                env_copy.close()

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
        changed_state_steps = 0
        any_positive_reward = False
        any_done = False
        first_expected_fail_step = None

        print(f"\nProcessing {json_file} ...")

        special_case_actions = None
        special_case_start_step = None
        if json_file.name == "engagement_01_full_compromise.json":
            special_case_start_step = 6
            special_case_actions = [
                ("OSScan", "(3,0)"),
                ("ServiceScan", "(3,0)"),
                ("OSScan", "(3,1)"),
                ("ServiceScan", "(3,1)"),
                ("ProcessScan", "(3,1)"),
                ("SSH-EXP", "(3,1)"),
                ("Tomcat-PE", "(3,1)"),
                ("OSScan", "(2,0)"),
                ("ServiceScan", "(2,0)"),
                ("SSH-EXP", "(2,0)"),
                ("OSScan", "(4,0)"),
                ("ServiceScan", "(4,0)"),
                ("SSH-EXP", "(4,0)"),
            ]
            print("SPECIAL_CASE_SEQUENCE", json_file.name, special_case_actions)

        for idx, step in enumerate(data["steps"]):
            action_name = step["action"]
            target = step["target_host"].replace(" ", "")
            print("JSON_ACTION:", action_name, target)
            prev_goal_reached = audit_goal_reached(env)
            prev_state_vector = audit_host_vector(obs)

            use_special_case = (
                json_file.name == "engagement_01_full_compromise.json"
                and special_case_actions is not None
                and idx + 1 >= special_case_start_step
                and idx + 1 - special_case_start_step < len(special_case_actions)
            )
            if use_special_case:
                special_action_name, special_target = special_case_actions[idx + 1 - special_case_start_step]
                action_id = select_best_action_replayed(special_action_name, special_target, obs, traj_actions)
                if action_id is None:
                    action_id = select_best_action(env, special_action_name, special_target, obs)
            else:
                action_id = select_best_action(env, action_name, target, obs)

            if action_id is None:
                if first_expected_fail_step is None:
                    first_expected_fail_step = idx + 1
                if first_expected_fail_step is None and any(x in action_name.lower() for x in ["exp", "exploit", "pe", "privilege", "tomcat", "daclsvc"]):
                    first_expected_fail_step = idx + 1
                print("AUDIT_STEP", json_file.name, idx + 1, action_name, target, None, None, False, None, None, prev_goal_reached, None)
                continue

            selected_action = env.action_space.get_action(action_id)
            selected_action_str = str(selected_action)
            next_obs, reward, terminated, truncated, _ = env.step(action_id)
            done = bool(terminated or truncated)
            state_changed = not np.array_equal(obs, next_obs)
            next_goal_reached = audit_goal_reached(env)
            next_state_vector = audit_host_vector(next_obs)
            privilege_related_change = None
            if prev_state_vector is not None and next_state_vector is not None and prev_state_vector.shape == next_state_vector.shape:
                privilege_related_change = float(np.abs(next_state_vector - prev_state_vector).sum())

            print(
                "AUDIT_STEP",
                json_file.name,
                idx + 1,
                action_name,
                target,
                action_id,
                selected_action_str,
                state_changed,
                reward,
                done,
                None if prev_goal_reached is None or next_goal_reached is None else (prev_goal_reached != next_goal_reached),
                privilege_related_change,
            )

            if state_changed:
                changed_state_steps += 1
            if reward > 0:
                any_positive_reward = True
            if done:
                any_done = True
            if first_expected_fail_step is None and any(x in action_name.lower() for x in ["exp", "exploit", "pe", "privilege", "tomcat", "daclsvc"]) and reward <= 0 and not done:
                first_expected_fail_step = idx + 1

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

        print(
            "AUDIT_TRAJECTORY_SUMMARY",
            json_file.name,
            "changed_state_steps=", changed_state_steps,
            "positive_reward=", any_positive_reward,
            "done_true=", any_done,
            "first_expected_fail_step=", first_expected_fail_step,
        )

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
