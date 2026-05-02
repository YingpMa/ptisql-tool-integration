import argparse
import glob
import json
import os
import random
from collections import Counter
from pathlib import Path

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


VALID_ACTIONS = {
    "scan_basic",
    "scan_service",
    "exploit_bindshell",
    "exploit_vsftpd",
    "exploit_unrealircd",
    "exploit_distccd",
    "exploit_samba",
    "stop",
}


ACTION_ALIASES = {
    "basic_scan": "scan_basic",
    "service_scan": "scan_service",
    "bindshell": "exploit_bindshell",
    "vsftpd": "exploit_vsftpd",
    "ftp": "exploit_vsftpd",
    "unrealircd": "exploit_unrealircd",
    "distccd": "exploit_distccd",
    "samba": "exploit_samba",
    "smb": "exploit_samba",
}


def normalize_action(action):
    if action is None:
        return None
    action = str(action).strip()
    return ACTION_ALIASES.get(action, action)


def state_to_vector(state):
    if state is None:
        state = {}

    vec = []
    for key in STATE_KEYS:
        value = state.get(key, 0)

        if isinstance(value, bool):
            vec.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            vec.append(float(value))
        else:
            vec.append(0.0)

    return vec


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def should_keep_run(data, policies, success_only):
    if policies:
        policy_name = data.get("policy_name", "")
        if policy_name not in policies:
            return False, "policy_filtered"

    if success_only and not bool(data.get("final_success", False)):
        return False, "failure_filtered"

    return True, None


def convert_one_run(path, use_executed_action=True, keep_unknown=False):
    """
    Convert one MSF batch trajectory file into replay transitions.

    Expected source structure:
      history[i]["state_before"] -> transition state
      history[i]["action_executed"] -> action
      history[i]["state_after"] -> next_state
      history[i]["reward"] -> reward
      history[i]["done"] -> done

    We use action_executed by default because action_requested can be
    high-level, e.g. auto_exploit, while the agent action space needs the
    concrete action, e.g. exploit_bindshell.
    """
    data = load_json(path)
    history = data.get("history", [])

    if not isinstance(history, list) or not history:
        return None, "empty_history"

    trajectory = []

    for step in history:
        if not isinstance(step, dict):
            continue

        if use_executed_action:
            action = normalize_action(step.get("action_executed"))
        else:
            action = normalize_action(step.get("action_requested"))

        if action not in VALID_ACTIONS:
            if keep_unknown:
                # Still skip for training because action_to_id would not contain it.
                pass
            continue

        state_before = step.get("state_before")
        state_after = step.get("state_after")

        if state_before is None or state_after is None:
            continue

        transition = {
            "state": state_to_vector(state_before),
            "action": action,
            "reward": float(step.get("reward", 0.0)),
            "next_state": state_to_vector(state_after),
            "done": bool(step.get("done", False)),
        }

        trajectory.append(transition)

    if not trajectory:
        return None, "empty_trajectory"

    # Make sure final transition is terminal if the log did not mark it.
    if not any(t["done"] for t in trajectory):
        trajectory[-1]["done"] = True

    metadata = {
        "source_file": str(path),
        "run_id": data.get("run_id", ""),
        "target": data.get("target", ""),
        "lhost": data.get("lhost", ""),
        "policy_name": data.get("policy_name", ""),
        "final_success": bool(data.get("final_success", False)),
        "used_metasploit": data.get("summary", {}).get("used_metasploit", None),
        "used_bindshell": data.get("summary", {}).get("used_bindshell", None),
        "num_steps": len(trajectory),
        "requested_actions": data.get("summary", {}).get("requested_actions", []),
        "executed_actions": data.get("summary", {}).get("executed_actions", []),
    }

    return {
        "trajectory": trajectory,
        "metadata": metadata,
    }, None


def main():
    parser = argparse.ArgumentParser(
        description="Convert MSF batch trajectory logs into IQ replay format."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="tool_integration/dataset/real_logs/msf_rl_env_runs",
        help="Directory containing msf_batch_*.json files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="tool_integration/outputs/replay_iq_msf.json",
        help="Output replay JSON path.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Input file pattern.",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=0,
        help="Maximum number of input files to use. 0 means all.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle input files before max_runs selection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--success_only",
        action="store_true",
        help="Only keep runs with final_success=true.",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="",
        help="Comma-separated policy names to keep, e.g. good,recover,noisy,fail. Empty means all.",
    )
    parser.add_argument(
        "--use_requested_action",
        action="store_true",
        help="Use action_requested instead of action_executed. Not recommended because auto_exploit is not in agent action space.",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)

    files = sorted(glob.glob(str(input_dir / args.pattern)))

    if args.shuffle:
        random.shuffle(files)

    if args.max_runs and args.max_runs > 0:
        files = files[: args.max_runs]

    policies = {p.strip() for p in args.policies.split(",") if p.strip()}

    trajectories = []
    runs = []

    skipped = Counter()
    action_counter = Counter()
    policy_counter = Counter()
    success_counter = Counter()

    for path in files:
        try:
            data = load_json(path)
        except Exception:
            skipped["json_error"] += 1
            continue

        keep, reason = should_keep_run(
            data=data,
            policies=policies,
            success_only=args.success_only,
        )
        if not keep:
            skipped[reason] += 1
            continue

        converted, error = convert_one_run(
            path,
            use_executed_action=not args.use_requested_action,
        )

        if error is not None:
            skipped[error] += 1
            continue

        traj = converted["trajectory"]
        meta = converted["metadata"]

        trajectories.append(traj)
        runs.append(meta)

        policy_counter[meta["policy_name"]] += 1
        success_counter["success" if meta["final_success"] else "fail"] += 1

        for step in traj:
            action_counter[step["action"]] += 1

    if not trajectories:
        print("=" * 80)
        print("[ERROR] No trajectories converted.")
        print(f"input_dir: {input_dir}")
        print(f"files found: {len(files)}")
        print(f"skipped: {dict(skipped)}")
        print("=" * 80)
        raise RuntimeError("No trajectories converted.")

    output = {
        "state_keys": STATE_KEYS,
        "num_trajectories": len(trajectories),
        "total_steps": sum(len(t) for t in trajectories),
        "action_distribution": dict(action_counter),
        "trajectories": trajectories,
        "metadata": {
            "source": "msf_tool_v3_verified_modules",
            "input_dir": str(input_dir),
            "num_input_files": len(files),
            "num_trajectories": len(trajectories),
            "num_transitions": sum(len(t) for t in trajectories),
            "policy_distribution": dict(policy_counter),
            "success_distribution": dict(success_counter),
            "success_ratio": success_counter["success"] / max(1, len(trajectories)),
            "action_distribution": dict(action_counter),
            "skipped": dict(skipped),
            "success_only": args.success_only,
            "policies": sorted(policies),
            "action_source": (
                "action_requested" if args.use_requested_action else "action_executed"
            ),
        },
        "runs": runs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("[OK] MSF batch logs converted to IQ replay")
    print("=" * 80)
    print(f"input_dir:          {input_dir}")
    print(f"files found:        {len(files)}")
    print(f"trajectories saved: {len(trajectories)}")
    print(f"total steps:        {output['total_steps']}")
    print(f"output_path:        {output_path}")
    print(f"policy dist:        {dict(policy_counter)}")
    print(f"success dist:       {dict(success_counter)}")
    print(f"success ratio:      {output['metadata']['success_ratio']:.3f}")
    print(f"action dist:        {dict(action_counter)}")
    print(f"skipped:            {dict(skipped)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
