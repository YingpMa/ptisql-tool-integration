import os
import glob
import json
from collections import Counter

LOG_DIR = "real_logs/rl_env_runs"

files = sorted(glob.glob(os.path.join(LOG_DIR, "*.json")))

total = 0
success = 0
fail = 0

lengths = []
success_lengths = []
fail_lengths = []

path_counter = Counter()
success_path_counter = Counter()
fail_path_counter = Counter()
transition_counter = Counter()
policy_counter = Counter()
action_counter = Counter()

for f in files:
    try:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception:
        continue

    history = data.get("history", [])
    if not history:
        continue

    actions = [step.get("action") for step in history if step.get("action")]
    if not actions:
        continue

    policy = data.get("policy_name", data.get("type", "unknown"))

    total += 1
    policy_counter[policy] += 1
    lengths.append(len(actions))
    action_counter.update(actions)

    path_key = tuple(actions)
    path_counter[path_key] += 1

    is_success = (
        policy in ["good", "recover"]
        or "exploit_bindshell" in actions
    ) and policy != "fail"

    if is_success:
        success += 1
        success_lengths.append(len(actions))
        success_path_counter[path_key] += 1
    else:
        fail += 1
        fail_lengths.append(len(actions))
        fail_path_counter[path_key] += 1

    for i in range(len(actions) - 1):
        transition_counter[(actions[i], actions[i + 1])] += 1


def avg(xs):
    return round(sum(xs) / len(xs), 2) if xs else "N/A"


print("=" * 80)
print("BASIC")
print("=" * 80)
print("total runs:", total)
print("success:", success)
print("fail:", fail)
print("success ratio:", round(success / total, 3) if total else 0)

print("\n" + "=" * 80)
print("LENGTH SUMMARY")
print("=" * 80)
print("all:", avg(lengths))
print("success:", avg(success_lengths))
print("fail:", avg(fail_lengths))

print("\n" + "=" * 80)
print("POLICY COUNTS")
print("=" * 80)
for k, v in policy_counter.most_common():
    print(f"{v:4d} | {k}")

print("\n" + "=" * 80)
print("ACTION COUNTS")
print("=" * 80)
for k, v in action_counter.most_common():
    print(f"{v:4d} | {k}")

print("\n" + "=" * 80)
print("TOP TRANSITIONS")
print("=" * 80)
for (a, b), c in transition_counter.most_common(20):
    print(f"{c:4d} | {a} -> {b}")

print("\n" + "=" * 80)
print("PATH DIVERSITY")
print("=" * 80)
print("Unique full paths:", len(path_counter))
print("Unique success paths:", len(success_path_counter))
print("Unique fail paths:", len(fail_path_counter))

top_success = success_path_counter.most_common(1)
top_fail = fail_path_counter.most_common(1)

print("Top success path share:", round(top_success[0][1] / success, 4) if success and top_success else 0)
print("Top fail path share:", round(top_fail[0][1] / fail, 4) if fail and top_fail else 0)
