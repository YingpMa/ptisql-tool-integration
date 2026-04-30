import os
import glob
import json
from collections import Counter

LOG_DIR = "dataset/real_logs/msf_rl_env_runs"
files = sorted(glob.glob(os.path.join(LOG_DIR, "*.json")))

fail_policy_counter = Counter()
fail_path_counter = Counter()
fail_cause_counter = Counter()
failed_exploit_counter = Counter()

for f in files:
    with open(f, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    if data.get("final_success", False):
        continue

    policy = data.get("policy_name", "unknown")
    history = data.get("history", [])

    executed = [
        step.get("action_executed")
        for step in history
        if step.get("action_executed")
    ]

    fail_policy_counter[policy] += 1
    fail_path_counter[tuple(executed)] += 1

    exploit_steps = [
        step for step in history
        if str(step.get("action_executed", "")).startswith("exploit_")
    ]

    failed_exploits = []
    successful_exploits = []

    for step in exploit_steps:
        action = step.get("action_executed")
        info = step.get("info", {})
        if info.get("success") is True:
            successful_exploits.append(action)
        else:
            failed_exploits.append(action)

    if not exploit_steps:
        fail_cause_counter["no_exploit_attempted"] += 1
    elif failed_exploits and not successful_exploits:
        fail_cause_counter["only_failed_exploit_attempts"] += 1
    elif successful_exploits:
        fail_cause_counter["unexpected_failed_final_after_success"] += 1
    else:
        fail_cause_counter["unknown"] += 1

    failed_exploit_counter.update(failed_exploits)

print("=" * 80)
print("FAILED RUNS BY POLICY")
print("=" * 80)
for k, v in fail_policy_counter.most_common():
    print(f"{v:4d} | {k}")

print("\n" + "=" * 80)
print("FAILED RUN CAUSES")
print("=" * 80)
for k, v in fail_cause_counter.most_common():
    print(f"{v:4d} | {k}")

print("\n" + "=" * 80)
print("FAILED EXPLOIT ACTIONS")
print("=" * 80)
for k, v in failed_exploit_counter.most_common():
    print(f"{v:4d} | {k}")

print("\n" + "=" * 80)
print("TOP FAILED PATHS")
print("=" * 80)
for path, count in fail_path_counter.most_common(20):
    print(f"{count:4d} | {' -> '.join(path)}")
