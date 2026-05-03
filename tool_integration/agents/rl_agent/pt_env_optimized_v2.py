"""
Optimized v2 environment.

Place this file at:
  tool_integration/agents/rl_agent/pt_env_optimized_v2.py

It reuses the existing optimized v1 environment and changes only the service
scan strategy:
  - v1: exploit-aware staged scan, but still uses -sV -sC on discovered+relevant ports
  - v2: lightweight service fingerprinting using -sV --version-light only on exploit-relevant ports

This keeps the comparison clean: baseline -> optimized v1 -> optimized v2.
"""

import json
from copy import deepcopy
from pathlib import Path

from tool_integration.agents.rl_agent.pt_env_optimized import (
    RealPTEnv as OptimizedV1Env,
)


class RealPTEnv(OptimizedV1Env):
    """
    Optimized v2 RealPTEnv.

    The original optimized v1 file already contains:
    - MetricsTracker
    - episode-local scan cache
    - exploit-aware basic scan
    - tool-level precondition filtering

    This v2 keeps all of that and only makes scan_service faster by avoiding
    default NSE scripts (-sC) and scanning only exploit-relevant ports.
    """

    EXPLOIT_RELEVANT_PORTS = {
        21,  # vsftpd
        139,  # samba
        445,  # samba
        1524,  # bindshell
        3632,  # distccd
        6667,  # unrealircd
        8180,  # tomcat
    }

    def __init__(
        self,
        target_ip="10.11.202.189",
        log_dir="tool_integration/dataset/real_logs/rl_env_runs_optimized_v2",
        use_metasploit=False,
        msf_password="msfpass",
        msf_host="127.0.0.1",
        msf_port=55552,
    ):
        super().__init__(
            target_ip=target_ip,
            log_dir=log_dir,
            use_metasploit=use_metasploit,
            msf_password=msf_password,
            msf_host=msf_host,
            msf_port=msf_port,
        )

    def _ports_to_csv_v2(self, ports):
        return ",".join(str(p) for p in sorted(set(int(p) for p in ports)))

    def _run_nmap_service(self):
        """
        Optimized v2 service scan.

        Main speedup:
        - scan only exploit-relevant ports
        - use -sV --version-light
        - avoid -sC, because default NSE scripts dominate latency

        This is still compatible with the current action space because the
        action space only contains exploits that depend on these ports.
        """
        ports = set(self.EXPLOIT_RELEVANT_PORTS)

        # If basic scan discovered relevant ports, keep them. Do not add all
        # discovered ports, otherwise the service scan becomes slow again.
        discovered_relevant = set(getattr(self, "last_open_ports", [])).intersection(
            self.EXPLOIT_RELEVANT_PORTS
        )
        ports.update(discovered_relevant)

        port_list = self._ports_to_csv_v2(ports)
        cache_key = ("scan_service_v2_light", self.target_ip, port_list)

        if cache_key in self.scan_cache:
            cached = deepcopy(self.scan_cache[cache_key])
            cached["from_cache"] = True
            cached["executed"] = False
            cached["duration"] = 0.0
            return cached

        cmd = [
            "nmap",
            "-Pn",
            "-T4",
            "-sV",
            "--version-light",
            "-p",
            port_list,
            self.target_ip,
        ]

        result, duration = self._run_timed_command(cmd, timeout=60)

        info = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "duration": duration,
            "executed": True,
            "from_cache": False,
            "port_list": port_list,
            "cmd": cmd,
            "scan_mode": "lightweight_service_fingerprinting_v2",
        }

        self.scan_cache[cache_key] = deepcopy(info)
        return info

    def _save_run(self):
        """Same as v1, but explicitly marks optimisation version as v2."""
        path = Path(self.log_dir) / f"{self.run_id}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": self.run_id,
                    "target": self.target_ip,
                    "use_metasploit": self.use_metasploit,
                    "optimized": True,
                    "optimization_version": "v2",
                    "optimizations": [
                        "metrics_logging",
                        "episode_local_scan_caching",
                        "exploit_aware_staged_scanning",
                        "lightweight_service_fingerprinting",
                        "tool_level_precondition_filtering",
                    ],
                    "state_keys": self.STATE_KEYS,
                    "actions": self.ACTIONS,
                    "metrics": self.metrics.summary(),
                    "history": self.history,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        return str(path)


if __name__ == "__main__":
    import random

    env = RealPTEnv(use_metasploit=False)
    obs = env.reset()
    done = False
    last_info = {}

    while not done:
        action = random.choice(list(env.ACTIONS.keys()))
        obs, reward, done, last_info = env.step(action)
        print(
            env.steps,
            env.ACTIONS[action],
            reward,
            last_info.get("status"),
            "executed=",
            last_info.get("executed"),
        )

    print("Saved:", last_info.get("saved_run_path"))
    print("Metrics:", last_info.get("metrics"))
