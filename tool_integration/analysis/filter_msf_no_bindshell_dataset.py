import json
import glob
import shutil
from collections import Counter
from pathlib import Path

SRC_DIR = "tool_integration/dataset/real_logs/msf_rl_env_runs"
OUT_DIR = "tool_integration/dataset/real_logs/msf_rl_env_runs_no_bindshell_success"

BIND_ACTION = "exploit_bindshell"

MSF_EXPLOITS = {
    "exploit_vsftpd",
    "exploit_samba",
    "exploit_unrealircd",
    "exploit_distccd",
}


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[SKIP] failed to read {path}: {e}")
        return None


def get_history(data):
    hist = data.get("history", [])
    return hist if isinstance(hist, list) else []


def get_requested_action(step):
    info = step.get("info", {}) or {}

    return (
        step.get("action_requested")
        or step.get("requested_action")
        or step.get("requested_action_name")
        or step.get("requested_pair_action_name")
        or info.get("action_requested")
        or info.get("requested_action")
        or info.get("requested_action_name")
        or info.get("requested_pair_action_name")
        or "unknown"
    )


def get_executed_action(step):
    info = step.get("info", {}) or {}

    return (
        step.get("action_executed")
        or step.get("executed_action")
        or step.get("executed_action_name")
        or step.get("action")
        or info.get("action_executed")
        or info.get("executed_action")
        or info.get("executed_action_name")
        or info.get("action")
        or "unknown"
    )


def get_backend_or_tool(step):
    info = step.get("info", {}) or {}

    value = (
        info.get("backend")
        or info.get("tool")
        or step.get("backend")
        or step.get("tool")
        or ""
    )

    return str(value).lower()


def is_success(data):
    summary = data.get("summary", {}) or {}

    if data.get("final_success") is True or data.get("final_success") == 1:
        return True

    if data.get("success") is True or data.get("success") == 1:
        return True

    if summary.get("final_success") is True or summary.get("final_success") == 1:
        return True

    if summary.get("final_has_shell") is True or summary.get("final_has_shell") == 1:
        return True

    if int(summary.get("num_successful_exploits", 0) or 0) > 0:
        return True

    return False


def classify_file(data):
    history = get_history(data)
    summary = data.get("summary", {}) or {}

    if not history:
        return False, "empty_history", [], [], []

    requested_actions = summary.get("requested_actions")
    executed_actions = summary.get("executed_actions")

    if not isinstance(requested_actions, list):
        requested_actions = [get_requested_action(step) for step in history]

    if not isinstance(executed_actions, list):
        executed_actions = [get_executed_action(step) for step in history]

    tools = [get_backend_or_tool(step) for step in history]

    full_text = json.dumps(data).lower()

    success = is_success(data)

    used_bindshell = (
        summary.get("used_bindshell") is True
        or BIND_ACTION in executed_actions
        or BIND_ACTION in requested_actions
        or "exploit_bindshell" in full_text
        or '"tool": "nc"' in full_text
    )

    used_metasploit = (
        summary.get("used_metasploit") is True
        or int(summary.get("num_metasploit_attempts", 0) or 0) > 0
        or "metasploit" in full_text
        or "msfrpc" in full_text
        or any(t in {"metasploit", "msf"} for t in tools)
    )

    has_msf_exploit_action = bool(set(executed_actions).intersection(MSF_EXPLOITS))
    has_scan_service = (
        "scan_service" in executed_actions or "scan_service" in requested_actions
    )

    if not success:
        return False, "not_success", requested_actions, executed_actions, tools

    if used_bindshell:
        return (
            False,
            "contains_bindshell_or_nc",
            requested_actions,
            executed_actions,
            tools,
        )

    if not has_scan_service:
        return False, "no_service_scan", requested_actions, executed_actions, tools

    if not has_msf_exploit_action:
        return (
            False,
            "no_msf_exploit_action",
            requested_actions,
            executed_actions,
            tools,
        )

    # 如果你的日志里 used_metasploit 记录准确，可以强制打开下面三行。
    # 但有些旧日志可能只记录 action，不记录 backend，所以先不强制。
    # if not used_metasploit:
    #     return False, "no_metasploit_backend", requested_actions, executed_actions, tools

    return True, "keep", requested_actions, executed_actions, tools


def main():
    src = Path(SRC_DIR)
    out = Path(OUT_DIR)

    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(src / "*.json")))

    total = 0
    kept = 0

    reason_counter = Counter()
    requested_counter_all = Counter()
    executed_counter_all = Counter()
    requested_counter_kept = Counter()
    executed_counter_kept = Counter()
    tool_counter_kept = Counter()
    kept_path_counter = Counter()

    kept_files = []

    for path in files:
        data = load_json(path)
        if data is None:
            continue

        keep, reason, requested_actions, executed_actions, tools = classify_file(data)

        total += 1
        reason_counter[reason] += 1

        for a in requested_actions:
            requested_counter_all[a] += 1

        for a in executed_actions:
            executed_counter_all[a] += 1

        if not keep:
            continue

        kept += 1

        for a in requested_actions:
            requested_counter_kept[a] += 1

        for a in executed_actions:
            executed_counter_kept[a] += 1

        for t in tools:
            if t:
                tool_counter_kept[t] += 1

        kept_path_counter[tuple(executed_actions)] += 1

        src_path = Path(path)
        dst_path = out / src_path.name
        shutil.copy2(src_path, dst_path)
        kept_files.append(str(dst_path))

    manifest_path = out / "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_dir": SRC_DIR,
                "output_dir": OUT_DIR,
                "filter": {
                    "success_only": True,
                    "exclude_bindshell": True,
                    "exclude_nc": True,
                    "require_scan_service": True,
                    "require_any_executed_msf_exploit": sorted(MSF_EXPLOITS),
                },
                "total_input": total,
                "kept": kept,
                "reasons": dict(reason_counter),
                "kept_files": kept_files,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 80)
    print("FILTER MSF-ONLY NO-BINDSHELL DATASET")
    print("=" * 80)
    print("source:", SRC_DIR)
    print("output:", OUT_DIR)
    print("total input:", total)
    print("kept:", kept)
    print("kept ratio:", round(kept / max(1, total), 4))

    print("\nREASONS")
    print("-" * 80)
    for k, v in reason_counter.most_common():
        print(f"{v:5d} | {k}")

    print("\nALL REQUESTED ACTIONS")
    print("-" * 80)
    for k, v in requested_counter_all.most_common():
        print(f"{v:5d} | {k}")

    print("\nALL EXECUTED ACTIONS")
    print("-" * 80)
    for k, v in executed_counter_all.most_common():
        print(f"{v:5d} | {k}")

    print("\nKEPT REQUESTED ACTIONS")
    print("-" * 80)
    for k, v in requested_counter_kept.most_common():
        print(f"{v:5d} | {k}")

    print("\nKEPT EXECUTED ACTIONS")
    print("-" * 80)
    for k, v in executed_counter_kept.most_common():
        print(f"{v:5d} | {k}")

    print("\nKEPT TOOLS / BACKENDS")
    print("-" * 80)
    for k, v in tool_counter_kept.most_common():
        print(f"{v:5d} | {k}")

    print("\nTOP KEPT EXECUTED PATHS")
    print("-" * 80)
    for path_tuple, v in kept_path_counter.most_common(20):
        print(f"{v:5d} | {' -> '.join(path_tuple)}")

    print("\nMANIFEST")
    print("-" * 80)
    print(manifest_path)

    if kept >= 100:
        print("\nOK: enough filtered MSF-only/no-bindshell trajectories for training.")
    elif kept >= 30:
        print("\nMAYBE: usable but small. Try training, but more data may help.")
    else:
        print(
            "\nWEAK: too few. You probably need to collect more MSF-only trajectories."
        )


if __name__ == "__main__":
    main()
