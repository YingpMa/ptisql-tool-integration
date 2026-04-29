import os
import glob
import json
from collections import Counter, defaultdict

LOG_DIR = "dataset/real_logs/msf_rl_env_runs"

files = sorted(glob.glob(os.path.join(LOG_DIR, "*.json")))[:650]

total = 0
success = 0
fail = 0

lengths = []
success_lengths = []
fail_lengths = []

policy_counter = Counter()
action_counter = Counter()
executed_action_counter = Counter()

path_counter = Counter()
success_path_counter = Counter()
fail_path_counter = Counter()

transition_counter = Counter()

# ⭐ 新增：exploit 成功率分析
exploit_attempts = Counter()
exploit_success = Counter()

for f in files:
    try:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except:
        continue

    history = data.get("history", [])
    if not history:
        continue

    # 👉 正确 success 判定
    is_success = data.get("final_success", False)

    policy = data.get("policy_name", "unknown")

    actions = []
    executed_actions = []

    for step in history:
        a = step.get("action_requested")
        e = step.get("action_executed")

        if a:
            actions.append(a)
            action_counter[a] += 1

        if e:
            executed_actions.append(e)
            executed_action_counter[e] += 1

        # ⭐ exploit 成功率统计
        if e and e.startswith("exploit_"):
            exploit_attempts[e] += 1
            if step.get("state_after", {}).get("has_shell"):
                exploit_success[e] += 1

    if not actions:
        continue

    total += 1
    policy_counter[policy] += 1
    lengths.append(len(actions))

    path_key = tuple(executed_actions)
    path_counter[path_key] += 1

    if is_success:
        success += 1
        success_lengths.append(len(actions))
        success_path_counter[path_key] += 1
    else:
        fail += 1
        fail_lengths.append(len(actions))
        fail_path_counter[path_key] += 1

    for i in range(len(executed_actions) - 1):
        transition_counter[(executed_actions[i], executed_actions[i + 1])] += 1


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
print("LENGTH")
print("=" * 80)
print("all:", avg(lengths))
print("success:", avg(success_lengths))
print("fail:", avg(fail_lengths))

print("\n" + "=" * 80)
print("POLICY")
print("=" * 80)
for k, v in policy_counter.most_common():
    print(f"{v:4d} | {k}")

print("\n" + "=" * 80)
print("ACTION (requested)")
print("=" * 80)
for k, v in action_counter.most_common():
    print(f"{v:4d} | {k}")

print("\n" + "=" * 80)
print("ACTION (executed)")
print("=" * 80)
for k, v in executed_action_counter.most_common():
    print(f"{v:4d} | {k}")

print("\n" + "=" * 80)
print("TRANSITIONS")
print("=" * 80)
for (a, b), c in transition_counter.most_common(20):
    print(f"{c:4d} | {a} -> {b}")

print("\n" + "=" * 80)
print("PATH DIVERSITY")
print("=" * 80)
print("Unique paths:", len(path_counter))
print("Unique success paths:", len(success_path_counter))
print("Unique fail paths:", len(fail_path_counter))

top_success = success_path_counter.most_common(1)
top_fail = fail_path_counter.most_common(1)

print("Top success path ratio:", round(top_success[0][1] / success, 4) if success else 0)
print("Top fail path ratio:", round(top_fail[0][1] / fail, 4) if fail else 0)

print("\n" + "=" * 80)
print("EXPLOIT SUCCESS RATE ⭐")
print("=" * 80)

for k in exploit_attempts:
    succ = exploit_success[k]
    tot = exploit_attempts[k]
    print(f"{k:20s} | {succ}/{tot} = {round(succ/tot, 3) if tot else 0}")
