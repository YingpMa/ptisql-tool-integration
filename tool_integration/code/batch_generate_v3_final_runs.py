import json
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

TARGET_IP = "10.11.202.189"
OUT_DIR = Path("real_logs/rl_env_runs")
N_RUNS = 650

EXPLOITS = [
    "exploit_bindshell",
    "exploit_vsftpd",
    "exploit_unrealircd",
    "exploit_distccd",
    "exploit_samba",
]


def now():
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
        "has_unrealircd": False,
        "has_distccd": False,
        "has_shell": False,
        "basic_scanned": False,
        "service_scanned": False,
        "failed_attempts": 0,
        "successful_exploits": 0,
    }


def apply_action(state, action, policy):
    done = False

    if action == "scan_basic":
        state["basic_scanned"] = True
        state["num_open_ports"] = 23
        state["has_ftp"] = True
        state["has_http"] = True
        state["has_mysql"] = True
        state["has_postgresql"] = True
        state["has_bindshell_1524"] = True
        state["has_samba"] = True
        state["has_tomcat"] = True
        state["has_distccd"] = True
        state["has_unrealircd"] = True
        return -0.05, done, {"scan_type": "basic"}

    if action == "scan_service":
        state["basic_scanned"] = True
        state["service_scanned"] = True
        state["num_open_ports"] = 30
        state["has_ftp"] = True
        state["has_ssh"] = True
        state["has_telnet"] = True
        state["has_http"] = True
        state["has_mysql"] = True
        state["has_postgresql"] = True
        state["has_bindshell_1524"] = True
        state["has_samba"] = True
        state["has_tomcat"] = True
        state["has_vsftpd_234"] = True
        state["has_unrealircd"] = True
        state["has_distccd"] = True
        return -0.08, done, {"scan_type": "service"}

    if action.startswith("exploit_"):
        if policy == "fail":
            state["failed_attempts"] += 1
            return -1.0, done, {"success": False}

        if action in ["exploit_vsftpd", "exploit_unrealircd", "exploit_distccd", "exploit_samba"]:
            if not state["service_scanned"]:
                state["failed_attempts"] += 1
                return -1.2, done, {"success": False, "reason": "service_not_verified"}

        if action == "exploit_bindshell":
            if not (state["basic_scanned"] or state["service_scanned"]):
                state["failed_attempts"] += 1
                return -1.5, done, {"success": False, "reason": "no_scan"}

        state["has_shell"] = True
        state["successful_exploits"] += 1

        reward_map = {
            "exploit_bindshell": 12.0,
            "exploit_vsftpd": 10.0,
            "exploit_unrealircd": 9.0,
            "exploit_distccd": 8.5,
            "exploit_samba": 8.0,
        }
        return reward_map.get(action, 8.0), done, {"success": True}

    if action == "stop":
        done = True
        if state["has_shell"]:
            return 6.0, done, {"stopped": True, "final_success": True}
        return -2.0, done, {"stopped": True, "final_success": False}

    return -1.0, done, {"error": "unknown_action"}


def record(history, step, action, state, reward, done, info):
    history.append({
        "step": step,
        "action": action,
        "state": deepcopy(state),
        "reward": float(reward),
        "done": bool(done),
        "info": info,
    })


def gen_good():
    return [
        random.choice(["scan_basic", "scan_service"]),
        random.choice(["scan_basic", "scan_service"]),
        random.choice(EXPLOITS),
        random.choice(EXPLOITS),
        "stop",
    ]


def gen_recover():
    return [
        random.choice(["scan_basic", "scan_service"]),
        random.choice(EXPLOITS),
        "scan_service",
        random.choice(EXPLOITS),
        "exploit_bindshell",
        "stop",
    ]


def gen_noisy():
    return [
        random.choice(["scan_basic", "scan_service"]),
        random.choice(["scan_basic", "scan_service"]),
        random.choice(["scan_basic", "scan_service"]),
        random.choice(EXPLOITS),
        random.choice(EXPLOITS),
        "stop",
    ]


def gen_fail():
    fail_paths = [
        ["scan_basic", "stop"],
        ["scan_service", "stop"],
        ["scan_basic", "scan_service", "stop"],
        ["scan_service", "scan_basic", "stop"],
        ["exploit_vsftpd", "scan_basic", "stop"],
        ["exploit_unrealircd", "scan_basic", "stop"],
        ["exploit_distccd", "scan_basic", "stop"],
        ["exploit_samba", "scan_basic", "stop"],
        ["exploit_bindshell", "scan_service", "stop"],
        ["scan_basic", "exploit_vsftpd", "stop"],
        ["scan_basic", "exploit_unrealircd", "stop"],
        ["scan_basic", "exploit_distccd", "stop"],
        ["scan_basic", "exploit_samba", "stop"],
        ["scan_basic", "scan_service", "scan_basic", "stop"],
        ["scan_service", "scan_basic", "scan_service", "stop"],
        ["scan_basic", "scan_service", "scan_basic", "scan_service", "stop"],
        ["scan_basic", "exploit_vsftpd", "exploit_bindshell", "stop"],
        ["scan_basic", "exploit_samba", "exploit_unrealircd", "stop"],
        ["scan_service", "exploit_samba", "exploit_distccd", "stop"],
        ["scan_service", "exploit_vsftpd", "exploit_samba", "stop"],
        ["scan_basic", "scan_basic", "exploit_vsftpd", "scan_service", "stop"],
        ["scan_service", "scan_basic", "exploit_distccd", "scan_basic", "stop"],
        ["scan_basic", "scan_service", "exploit_unrealircd", "scan_service", "stop"],
    ]
    return random.choice(fail_paths)


def rollout(actions, policy):
    state = empty_state()
    history = []
    done = False

    for i, action in enumerate(actions, start=1):
        if done:
            break
        reward, done, info = apply_action(state, action, policy)
        record(history, i, action, state, reward, done, info)

    if not done:
        reward, done, info = apply_action(state, "stop", policy)
        record(history, len(history) + 1, "stop", state, reward, done, info)

    return history


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    policies = (
        ["good"] * 90 +
        ["recover"] * 90 +
        ["noisy"] * 90 +
        ["fail"] * 80
    )

    plan = []
    while len(plan) < N_RUNS:
        plan.extend(policies)
    plan = plan[:N_RUNS]
    random.shuffle(plan)

    for idx, policy in enumerate(plan):
        if policy == "good":
            actions = gen_good()
        elif policy == "recover":
            actions = gen_recover()
        elif policy == "noisy":
            actions = gen_noisy()
        else:
            actions = gen_fail()

        history = rollout(actions, policy)

        out = {
            "run_id": f"batch_v3_{idx:03d}_{now()}",
            "target": TARGET_IP,
            "policy_name": policy,
            "action_plan": actions,
            "history": history,
        }

        path = OUT_DIR / f"{out['run_id']}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        print(f"[OK] {path.name} {policy} {actions}")


if __name__ == "__main__":
    main()
