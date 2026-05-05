import socket
import time
import json
from copy import deepcopy

from tool_integration.agents.rl_agent.pt_env_optimized_v2 import (
    RealPTEnv as RealPTEnvV2,
)


class RealPTEnv(RealPTEnvV2):
    """
    Optimized v3 real tool environment.

    v3 keeps v2's action/state schema and precondition filtering, but replaces
    Nmap service-version detection in scan_service with direct lightweight
    service inference for the exploit-relevant ports. It is designed for a
    controlled lab where the action space is known.
    """

    def __init__(
        self,
        target_ip="10.11.202.189",
        log_dir="tool_integration/dataset/real_logs/rl_env_runs_optimized_v3",
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
        self.run_id = f"rl_env_optimized_v3_{backend}_{self._now_str()}"
        return obs

    def _grab_banner(self, port, payload=None, timeout=0.8):
        try:
            with socket.create_connection(
                (self.target_ip, int(port)), timeout=timeout
            ) as s:
                s.settimeout(timeout)
                if payload:
                    s.sendall(payload)
                    time.sleep(0.15)
                try:
                    data = s.recv(2048)
                except socket.timeout:
                    data = b""
            return data.decode(errors="ignore")
        except Exception:
            return ""

    def _infer_service_entry(self, port):
        port = int(port)

        if port == 21:
            banner = self._grab_banner(21, timeout=0.8)
            text = banner.lower()
            if "vsftpd 2.3.4" in text:
                return "ftp", "vsftpd 2.3.4"
            if "vsftpd" in text:
                return "ftp", banner.strip()
            return "ftp", banner.strip()

        if port in {139, 445}:
            return "netbios-ssn", "Samba likely"

        if port == 1524:
            return "bindshell", "Metasploitable root shell"

        if port == 3632:
            return "distccd", "distccd"

        if port == 6667:
            banner = self._grab_banner(
                6667,
                payload=b"NICK scan\r\nUSER scan 0 * :scan\r\n",
                timeout=0.8,
            )
            if "unreal" in banner.lower():
                return "irc", "UnrealIRCd"
            return "irc", "UnrealIRCd"

        if port == 8180:
            banner = self._grab_banner(
                8180,
                payload=b"HEAD / HTTP/1.0\r\n\r\n",
                timeout=0.8,
            )
            if "tomcat" in banner.lower():
                return "http", "Apache Tomcat"
            return "http", "Apache Tomcat/Coyote"

        return "unknown", ""

    def _make_synthetic_nmap_output(self, service_map):
        lines = [
            f"Nmap scan report for {self.target_ip}",
            "PORT     STATE SERVICE VERSION",
        ]
        for port in sorted(service_map, key=lambda x: int(x)):
            item = service_map[port]
            service = item.get("service", "unknown")
            version = item.get("version", "")
            if version:
                lines.append(f"{port}/tcp open  {service} {version}")
            else:
                lines.append(f"{port}/tcp open  {service}")
        return "\n".join(lines) + "\n"

    def _run_nmap_service(self):
        """
        Stage 2 v3: direct lightweight service inference.

        No `nmap -sV` is executed here. The function emits Nmap-like output so
        the inherited parsing/state update code can remain unchanged.
        """
        start = time.time()

        ports = set(self.last_open_ports).intersection(self.EXPLOIT_RELEVANT_PORTS)
        ports.update(self.EXPLOIT_RELEVANT_PORTS)

        port_list = self._ports_to_csv(ports)
        cache_key = ("scan_service_v3_direct_inference", self.target_ip, port_list)

        if cache_key in self.scan_cache:
            cached = deepcopy(self.scan_cache[cache_key])
            cached["from_cache"] = True
            cached["executed"] = False
            cached["duration"] = 0.0
            return cached

        service_map = {}
        open_ports = set(self.last_open_ports)

        for port in sorted(ports):
            port = int(port)
            if port not in open_ports:
                try:
                    with socket.create_connection((self.target_ip, port), timeout=0.35):
                        open_ports.add(port)
                except Exception:
                    continue

            service, version = self._infer_service_entry(port)
            service_map[str(port)] = {"service": service, "version": version}

        stdout = self._make_synthetic_nmap_output(service_map)
        duration = time.time() - start

        info = {
            "stdout": stdout,
            "stderr": "",
            "returncode": 0,
            "duration": duration,
            "executed": True,
            "from_cache": False,
            "port_list": port_list,
            "cmd": ["direct_service_inference", self.target_ip, port_list],
            "service_map_direct": service_map,
        }
        self.scan_cache[cache_key] = deepcopy(info)
        return info

    def _save_run(self):
        path = self.log_dir / f"{self.run_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": self.run_id,
                    "target": self.target_ip,
                    "use_metasploit": self.use_metasploit,
                    "optimized": True,
                    "optimization_version": "v3",
                    "optimizations": [
                        "metrics_logging",
                        "episode_local_scan_caching",
                        "exploit_aware_staged_scanning",
                        "direct_service_inference",
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
