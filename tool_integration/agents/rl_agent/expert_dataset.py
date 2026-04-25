import json
import os
import glob
import numpy as np

ACTION_TO_ID = {
    "scan_basic": 0,
    "scan_service": 1,
    "exploit_bindshell": 2,
    "exploit_vsftpd": 3,
    "exploit_unrealircd": 4,
    "exploit_distccd": 5,
    "exploit_samba": 6,
    "stop": 7,
}


def state_to_vector(state):
    return np.asarray([
        float(state.get("num_open_ports", 0)),
        float(state.get("has_ftp", False)),
        float(state.get("has_ssh", False)),
        float(state.get("has_telnet", False)),
        float(state.get("has_http", False)),
        float(state.get("has_mysql", False)),
        float(state.get("has_postgresql", False)),
        float(state.get("has_bindshell_1524", False)),
        float(state.get("has_samba", False)),
        float(state.get("has_tomcat", False)),
        float(state.get("has_vsftpd_234", False)),
        float(state.get("has_unrealircd", False)),
        float(state.get("has_distccd", False)),
        float(state.get("has_shell", False)),
        float(state.get("basic_scanned", False)),
        float(state.get("service_scanned", False)),
        float(state.get("failed_attempts", 0)),
        float(state.get("successful_exploits", 0)),
    ], dtype=np.float32)


def empty_state():
    return {
        "num_open_ports": 0,
        "has_ftp": False,
        "has_ssh": False,
        "has_telnet": False,
        "has_http": False,
        "has_mysql": False,
        "has_postgresql": False,
        "has_bindshell_1524": False,
        "has_samba": False,
        "has_tomcat": False,
        "has_vsftpd_234": False,
        "has_unrealircd": False,
        "has_distccd": False,
        "has_shell": False,
        "basic_scanned": False,
        "service_scanned": False,
        "failed_attempts": 0,
        "successful_exploits": 0,
    }


def load_single_run_file(file_path):
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"Empty file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history")
    if not isinstance(history, list):
        raise ValueError(f"Unsupported format: {file_path}")

    transitions = []
    prev_state = empty_state()

    for item in history:
        action_name = item.get("action")
        if action_name not in ACTION_TO_ID:
            continue
        curr_state = item.get("state")
        if not isinstance(curr_state, dict):
            continue

        obs = state_to_vector(prev_state)
        next_obs = state_to_vector(curr_state)
        action = ACTION_TO_ID[action_name]
        reward = float(item.get("reward", 0.0))
        done = float(item.get("done", False))

        transitions.append((obs, action, reward, next_obs, done))
        prev_state = curr_state

    return transitions


def load_expert_transitions(expert_dir):
    files = sorted(glob.glob(os.path.join(expert_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No json files found in {expert_dir}")

    all_transitions = []
    skipped = []

    for fp in files:
        try:
            all_transitions.extend(load_single_run_file(fp))
        except Exception as e:
            skipped.append((fp, str(e)))

    if skipped:
        print("Skipped files:")
        for fp, reason in skipped[:10]:
            print(f"  - {fp}: {reason}")

    if not all_transitions:
        raise ValueError("Loaded 0 expert transitions")

    obs_dim = len(all_transitions[0][0])
    return all_transitions, obs_dim
