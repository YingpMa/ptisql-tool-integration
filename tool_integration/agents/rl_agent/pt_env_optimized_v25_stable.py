from tool_integration.agents.rl_agent.pt_env_optimized_v25 import (
    RealPTEnv as RealPTEnvV25,
)


class RealPTEnv(RealPTEnvV25):
    """
    Optimized v2.5-stable real tool-based PT environment.

    This environment keeps the same execution logic as v2.5:
    - Nmap-based basic scan with -n
    - lightweight Nmap service fingerprinting with -sV --version-light
    - no direct service inference
    - same tool-level precondition filtering as v2/v2.5

    The stability improvement is mainly implemented in the training entry point:
    - fast-path exploit_bindshell is allowed only as the first exploit attempt
    - after one failed attempt, the agent is forced back to scan_service

    This file exists separately so logs are saved to a separate directory and
    can be analysed independently from v2.5.
    """

    def __init__(
        self,
        target_ip="10.11.202.189",
        log_dir="tool_integration/dataset/real_logs/rl_env_runs_optimized_v25_stable",
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
        self.run_id = f"rl_env_optimized_v25_stable_{backend}_{self._now_str()}"
        return obs

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
                    "optimization_version": "v2.5-stable",
                    "optimizations": [
                        "metrics_logging",
                        "episode_local_scan_caching",
                        "exploit_aware_staged_scanning",
                        "lightweight_service_fingerprinting",
                        "nmap_dns_disabled",
                        "fast_path_bindshell_once",
                        "fallback_to_service_scan_after_fast_path_failure",
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
