import json
import os
import glob
import shutil
from collections import Counter
from pathlib import Path

SRC_DIR = "tool_integration/dataset/real_logs/msf_rl_env_runs"
OUT_DIR = "tool_integration/dataset/real_logs/msf_rl_env_runs_no_bindshell_success"

MSF_EXPLOITS = {
    "exploit_vsftpd",
    "exploit_samba",
    "exploit_unrealircd",
    "exploit_distccd",
}

BIND_ACTION = "exploit_bindshell"


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


def get_action(step):
    info = step.get("info", {}) or {}

    return (
        step.get("action")
        or step.get("executed_action_name")
        or info.get("executed_action_name")
        or info.get("requested_pair_action_name")
        or info.get("action")
        or "unknown"
    )


def get_backend(step):
    info = step.get("info", {}) or {}
    return info.get("backend") or step.get("backend") or ""


def is_success(data, history):
    metrics = data.get("metrics", {}) or {}

    if int(metrics.get("success", 0)) == 1:
        return True

    if data.get("success") is True or data.get("success") == 1:
        return True

    if data.get("goal_reached") is True or data.get("goal_reached") == 1:
        return True

    for step in history:
        info = step.get("info", {}) or {}

        if info.get("success") is True or info.get("success") == 1:
            return True

        if info.get("real_success") is True or info.get("real_success") == 1:
            return True

        if info.get("status") == "success" and get_action(step).startswith("exploit_"):
            return True

    return False


def should_keep(data):
    history = get_history(data)
    if not history:
        return False, "empty_history"

    actions = [get_action(step) for step in history]
    action_set = set(actions)

    success = is_success(data, history)

    if not success:
        return False, "not_success"

    if BIND_ACTION in action_set:
        return False, "contains_bindshell"

    if not action_set.intersection(MSF_EXPLOITS):
        return False, "no_msf_exploit_action"

    if "scan_service" not in action_set:
        return False, "no_service_scan"

    return True, "keep"


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
    action_counter_all = Counter()
    action_counter_kept = Counter()
    backend_counter_kept = Counter()
    path_counter_kept = Counter()

    kept_files = []

    for path in files:
        data = load_json(path)
        if data is None:
            continue

        history = get_history(data)
        if not history:
            reason_counter["empty_history"] += 1
            continue

        total += 1

        actions = [get_action(step) for step in history]
        for a in actions:
            action_counter_all[a] += 1

        keep, reason = should_keep(data)
        reason_counter[reason] += 1

        if not keep:
            continue

        kept += 1

        for a in actions:
            action_counter_kept[a] += 1

        for step in history:
            backend = get_backend(step)
            if backend:
                backend_counter_kept[backend] += 1

        path_counter_kept[tuple(actions)] += 1

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
                    "exclude_action": BIND_ACTION,
                    "require_any_action": sorted(MSF_EXPLOITS),
                    "require_scan_service": True,
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
    print("total usable input:", total)
    print("kept:", kept)
    print("kept ratio:", round(kept / max(1, total), 4))

    print("\nREASONS")
    print("-" * 80)
    for k, v in reason_counter.most_common():
        print(f"{v:5d} | {k}")

    print("\nALL ACTIONS")
    print("-" * 80)
    for k, v in action_counter_all.most_common():
        print(f"{v:5d} | {k}")

    print("\nKEPT ACTIONS")
    print("-" * 80)
    for k, v in action_counter_kept.most_common():
        print(f"{v:5d} | {k}")

    print("\nKEPT BACKENDS")
    print("-" * 80)
    for k, v in backend_counter_kept.most_common():
        print(f"{v:5d} | {k}")

    print("\nTOP KEPT PATHS")
    print("-" * 80)
    for path_tuple, v in path_counter_kept.most_common(20):
        print(f"{v:5d} | {' -> '.join(path_tuple)}")

    print("\nMANIFEST")
    print("-" * 80)
    print(manifest_path)

    if kept >= 100:
        print(
            "\nOK: enough filtered MSF-only successful trajectories for a first training attempt."
        )
    elif kept >= 30:
        print("\nMAYBE: usable but small. Try training, but may need more data.")
    else:
        print(
            "\nWEAK: probably too few. Consider collecting more MSF-only trajectories."
        )


if __name__ == "__main__":
    main()
