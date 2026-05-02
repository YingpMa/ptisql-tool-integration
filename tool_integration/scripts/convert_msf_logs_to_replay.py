import argparse
import glob
import json
import random
from collections import Counter
from pathlib import Path

DEFAULT_STATE_KEYS = [
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


def empty_state_dict(state_keys):
    state = {}

    for key in state_keys:
        if key in {
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
        }:
            state[key] = False
        else:
            state[key] = 0

    return state


def state_to_vector(state, state_keys):
    """
    Convert either dict state or already-vector state into the fixed vector form.
    """
    if state is None:
        state = empty_state_dict(state_keys)

    if isinstance(state, list):
        if len(state) != len(state_keys):
            raise ValueError(
                f"Vector state length mismatch: got {len(state)}, expected {len(state_keys)}"
            )
        return [float(x) for x in state]

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported state type: {type(state)}")

    return [float(state.get(k, 0.0)) for k in state_keys]


def infer_initial_state(data, state_keys):
    """
    Prefer an explicit initial_state if the log has one.
    Otherwise use an empty state.

    The current RealPTEnv logs usually store only post-step states in history,
    so the first transition state is reconstructed as the empty initial state.
    """
    for key in ["initial_state", "start_state", "reset_state"]:
        if key in data:
            return data[key]

    return empty_state_dict(state_keys)


def normalize_action(action):
    if action is None:
        return None

    action = str(action).strip()

    # Common aliases just in case old logs contain slightly different names.
    aliases = {
        "basic_scan": "scan_basic",
        "service_scan": "scan_service",
        "exploit_ftp": "exploit_vsftpd",
        "vsftpd": "exploit_vsftpd",
        "samba": "exploit_samba",
        "unrealircd": "exploit_unrealircd",
        "distccd": "exploit_distccd",
        "bindshell": "exploit_bindshell",
    }

    return aliases.get(action, action)


def is_success_run(data, trajectory):
    """
    A run is considered successful if:
    - top-level metrics.success == 1, or
    - any transition next_state has has_shell=True, or
    - final transition done=True and next_state has has_shell=True.
    """
    metrics = data.get("metrics", {})

    if isinstance(metrics, dict) and int(metrics.get("success", 0)) == 1:
        return True

    for step in trajectory:
        next_state = step.get("next_state_dict", {})
        if isinstance(next_state, dict) and bool(next_state.get("has_shell", False)):
            return True

    return False


def convert_one_file(path, state_keys, keep_invalid=False):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history", [])

    if not isinstance(history, list) or not history:
        return None, "empty_history"

    trajectory = []
    prev_state = infer_initial_state(data, state_keys)

    for item in history:
        if not isinstance(item, dict):
            continue

        action = normalize_action(item.get("action"))

        if action not in VALID_ACTIONS:
            continue

        info = item.get("info", {}) or {}
        status = info.get("status")

        # Optional: skip blocked invalid actions if you do not want the policy
        # to learn impossible tool calls.
        if not keep_invalid and status == "invalid":
            prev_state = item.get("state", prev_state)
            continue

        next_state = item.get("state")

        if next_state is None:
            # If a future logger stores next_state explicitly, support it.
            next_state = item.get("next_state")

        if next_state is None:
            continue

        reward = float(item.get("reward", 0.0))
        done = bool(item.get("done", False))

        transition = {
            "state": state_to_vector(prev_state, state_keys),
            "next_state": state_to_vector(next_state, state_keys),
            "action": action,
            "reward": reward,
            "done": done,
            # Debug metadata. Training ignores these fields.
            "source_file": str(path),
            "step": item.get("step"),
            "status": status,
        }

        # Keep dict form temporarily for success filtering/debugging.
        transition["next_state_dict"] = (
            next_state if isinstance(next_state, dict) else {}
        )

        trajectory.append(transition)
        prev_state = next_state

    if not trajectory:
        return None, "empty_trajectory"

    # Clean temporary dict before output.
    for step in trajectory:
        step.pop("next_state_dict", None)

    return {
        "trajectory": trajectory,
        "metadata": {
            "source_file": str(path),
            "run_id": data.get("run_id", ""),
            "target": data.get("target", ""),
            "use_metasploit": data.get("use_metasploit", None),
            "optimized": data.get("optimized", False),
            "num_steps": len(trajectory),
        },
    }, None


def trajectory_success_from_output(converted):
    for step in converted["trajectory"]:
        if step["next_state"][DEFAULT_STATE_KEYS.index("has_shell")] == 1.0:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert RealPTEnv / Metasploit run logs into IQ replay format."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="tool_integration/dataset/real_logs/msf_rl_env_runs",
        help="Directory containing msf run JSON logs.",
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
        help="Input JSON pattern. Default: *.json",
    )

    parser.add_argument(
        "--max_runs",
        type=int,
        default=0,
        help="Maximum number of runs to convert. 0 means all.",
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle input files before selecting max_runs.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--success_only",
        action="store_true",
        help="Keep only successful trajectories.",
    )

    parser.add_argument(
        "--keep_invalid",
        action="store_true",
        help="Keep invalid blocked actions as training transitions.",
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

    if not files:
        raise FileNotFoundError(
            f"No JSON files found in {input_dir} with pattern {args.pattern}"
        )

    state_keys = DEFAULT_STATE_KEYS

    trajectories = []
    run_metadata = []
    skipped = Counter()
    action_counter = Counter()
    status_counter = Counter()
    success_count = 0
    fail_count = 0

    for file in files:
        converted, error = convert_one_file(
            file,
            state_keys=state_keys,
            keep_invalid=args.keep_invalid,
        )

        if error is not None:
            skipped[error] += 1
            continue

        success = False

        # Check success from final vectors.
        has_shell_idx = state_keys.index("has_shell")
        for step in converted["trajectory"]:
            if step["next_state"][has_shell_idx] == 1.0:
                success = True
                break

        if args.success_only and not success:
            skipped["filtered_failure"] += 1
            continue

        for step in converted["trajectory"]:
            action_counter[step["action"]] += 1
            status_counter[str(step.get("status"))] += 1

            # Remove debug fields that are not needed by training.
            step.pop("source_file", None)
            step.pop("step", None)
            step.pop("status", None)

        trajectories.append(converted["trajectory"])

        metadata = converted["metadata"]
        metadata["success"] = success
        run_metadata.append(metadata)

        if success:
            success_count += 1
        else:
            fail_count += 1

    if not trajectories:
        raise RuntimeError("No trajectories converted. Check input format or filters.")

    output = {
        "state_keys": state_keys,
        "trajectories": trajectories,
        "metadata": {
            "source": "msf_real_env_runs",
            "input_dir": str(input_dir),
            "num_input_files": len(files),
            "num_trajectories": len(trajectories),
            "num_transitions": sum(len(t) for t in trajectories),
            "success_runs": success_count,
            "failed_runs": fail_count,
            "success_ratio": success_count / max(1, len(trajectories)),
            "action_distribution": dict(action_counter),
            "status_distribution": dict(status_counter),
            "skipped": dict(skipped),
            "keep_invalid": args.keep_invalid,
            "success_only": args.success_only,
        },
        "runs": run_metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("MSF LOGS -> IQ REPLAY CONVERSION")
    print("=" * 80)
    print(f"input_dir:        {input_dir}")
    print(f"output_path:      {output_path}")
    print(f"input files:      {len(files)}")
    print(f"trajectories:     {len(trajectories)}")
    print(f"transitions:      {output['metadata']['num_transitions']}")
    print(f"success runs:     {success_count}")
    print(f"failed runs:      {fail_count}")
    print(f"success ratio:    {output['metadata']['success_ratio']:.3f}")
    print(f"action dist:      {dict(action_counter)}")
    print(f"status dist:      {dict(status_counter)}")
    print(f"skipped:          {dict(skipped)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
