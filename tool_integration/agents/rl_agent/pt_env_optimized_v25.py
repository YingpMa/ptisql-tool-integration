from copy import deepcopy

from tool_integration.agents.rl_agent.pt_env_optimized_v2 import (
    RealPTEnv as RealPTEnvV2,
)


class RealPTEnv(RealPTEnvV2):
    """
    Optimized v2.5 real tool-based PT environment.

    v2.5 keeps the real Nmap + Metasploit workflow, unlike v3 direct inference.

    Compared with v2:
    1. Adds -n to Nmap commands to skip DNS resolution.
    2. Keeps exploit-aware staged scanning.
    3. Keeps lightweight service fingerprinting with -sV --version-light.
    4. Uses a separate log directory and run_id so results do not mix with v2.

    The fast-path exploit behaviour is mainly implemented in the training script:
    if scan_basic finds port 1524, the agent is allowed to choose exploit_bindshell
    before scan_service.
    """

    def __init__(
        self,
        target_ip="10.11.202.189",
        log_dir="tool_integration/dataset/real_logs/rl_env_runs_optimized_v25",
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

    def reset(self):
        obs = super().reset()
        backend = "msf" if self.use_metasploit else "script"
        self.run_id = f"rl_env_optimized_v25_{backend}_{self._now_str()}"
        return obs

    def _run_nmap_basic(self):
        """
        Stage 1: exploit-aware lightweight port scan with DNS lookup disabled.

        Compared with v2:
            nmap -Pn -T4 -p ...
        v2.5:
            nmap -n -Pn -T4 -p ...
        """
        port_list = self._ports_to_csv(self.KEY_PORTS)
        cache_key = ("scan_basic_v25", self.target_ip, port_list)

        if cache_key in self.scan_cache:
            cached = deepcopy(self.scan_cache[cache_key])
            cached["from_cache"] = True
            cached["executed"] = False
            cached["duration"] = 0.0
            return cached

        cmd = ["nmap", "-n", "-Pn", "-T4", "-p", port_list, self.target_ip]
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
        }

        self.scan_cache[cache_key] = deepcopy(info)
        return info

    def _run_nmap_service(self):
        """
        Stage 2: lightweight exploit-aware service fingerprinting with DNS disabled.

        This keeps the v2 real-tool approach:
            nmap -sV --version-light

        But adds -n:
            nmap -n -Pn -T4 -sV --version-light -p exploit-relevant target
        """
        ports = set(self.EXPLOIT_RELEVANT_PORTS)

        discovered_relevant = set(self.last_open_ports).intersection(
            self.EXPLOIT_RELEVANT_PORTS
        )
        ports.update(discovered_relevant)

        port_list = self._ports_to_csv(ports)
        cache_key = ("scan_service_v25_light", self.target_ip, port_list)

        if cache_key in self.scan_cache:
            cached = deepcopy(self.scan_cache[cache_key])
            cached["from_cache"] = True
            cached["executed"] = False
            cached["duration"] = 0.0
            return cached

        cmd = [
            "nmap",
            "-n",
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
        }

        self.scan_cache[cache_key] = deepcopy(info)
        return info

    def _save_run(self):
        path = self.log_dir / f"{self.run_id}.json"

        with open(path, "w", encoding="utf-8") as f:
            import json

            json.dump(
                {
                    "run_id": self.run_id,
                    "target": self.target_ip,
                    "use_metasploit": self.use_metasploit,
                    "optimized": True,
                    "optimization_version": "v2.5",
                    "optimizations": [
                        "metrics_logging",
                        "episode_local_scan_caching",
                        "exploit_aware_staged_scanning",
                        "lightweight_service_fingerprinting",
                        "nmap_dns_disabled",
                        "fast_path_bindshell_action_mask",
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
