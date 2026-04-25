import glob
import json
import os
import random
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

ID_TO_ACTION = {v: k for k, v in ACTION_TO_ID.items()}


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


class ReplayPTEnv:
  def __init__(self, log_dir="real_logs/rl_env_runs", mismatch_penalty=-1.0, seed=42):
    self.log_dir = log_dir
    self.mismatch_penalty = mismatch_penalty
    self.rng = random.Random(seed)
    self.episodes = self._load_episodes(log_dir)
    if not self.episodes:
      raise ValueError(f"No usable replay episodes found in {log_dir}")
    self.current_episode = None
    self.ptr = 0
    self.done = False

  def _load_episodes(self, log_dir):
    files = sorted(glob.glob(os.path.join(log_dir, "*.json")))
    episodes = []

    for fp in files:
      if os.path.getsize(fp) == 0:
        continue
      try:
        with open(fp, "r", encoding="utf-8") as f:
          data = json.load(f)
      except Exception:
        continue

      history = data.get("history")
      if not isinstance(history, list) or not history:
        continue

      prev_state = empty_state()
      transitions = []

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
        done = bool(item.get("done", False))
        transitions.append((obs, action, reward, next_obs, done, action_name, curr_state))
        prev_state = curr_state

      if transitions:
        episodes.append(transitions)

    return episodes

  def reset(self):
    self.current_episode = self.rng.choice(self.episodes)
    self.ptr = 0
    self.done = False
    return np.asarray(self.current_episode[0][0], dtype=np.float32)

  def _shape_reward(self, action_name, raw_reward, curr_state, matched):
    if not matched:
      return float(self.mismatch_penalty)

    if action_name == "scan_basic":
      return -0.03
    if action_name == "scan_service":
      return -0.05
    if action_name.startswith("exploit_"):
      if curr_state.get("has_shell", False) or raw_reward > 0:
        return max(raw_reward, 8.0)
      return -1.0
    if action_name == "stop":
      return 6.0 if curr_state.get("has_shell", False) or raw_reward > 0 else -2.0
    return raw_reward

  def step(self, action):
    if self.done:
      raise RuntimeError("Episode already done. Call reset().")

    if self.ptr >= len(self.current_episode):
      self.done = True
      return np.asarray(self.current_episode[-1][3], dtype=np.float32), 0.0, True, {
        "matched": False,
        "reason": "episode_exhausted",
      }

    obs, recorded_action, raw_reward, next_obs, recorded_done, action_name, curr_state = self.current_episode[self.ptr]
    matched = int(action) == int(recorded_action)
    reward = self._shape_reward(action_name, raw_reward, curr_state, matched)

    self.ptr += 1
    self.done = bool(recorded_done) or (self.ptr >= len(self.current_episode))

    return np.asarray(next_obs, dtype=np.float32), float(reward), self.done, {
      "matched": matched,
      "recorded_action": int(recorded_action),
      "chosen_action": int(action),
      "recorded_action_name": action_name,
      "ptr": self.ptr,
      "episode_len": len(self.current_episode),
    }
