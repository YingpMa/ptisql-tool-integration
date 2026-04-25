import json
import subprocess
from pathlib import Path
from collections import defaultdict


RUN_TIMES = 10
RUN_SCRIPT = "real_logs/run_agent.py"
RUN_DIR = Path("real_logs/agent_runs")


def run_agent_n_times(n):
    print(f"[*] Running agent {n} times...\n")
    for i in range(n):
        print(f"[{i+1}/{n}] Running...")
        subprocess.run(["python", RUN_SCRIPT])


def load_latest_runs(n):
    files = sorted(RUN_DIR.glob("agent_run_*.json"))
    return files[-n:]


def analyze_runs(files):
    action_counts = defaultdict(int)
    success_counts = defaultdict(int)
    total_reward = defaultdict(float)

    results = []

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        step1 = data["steps"][1]
        step2 = data["steps"][2]

        action = step1["result"]["predicted_action"]
        success = step2["result"].get("success", False)
        real_success = step2["result"].get("real_success", False)
        reward = step2["reward"]

        action_counts[action] += 1
        if success:
            success_counts[action] += 1

        total_reward[action] += reward

        results.append({
            "file": path.name,
            "action": action,
            "success": success,
            "real_success": real_success,
            "reward": reward
        })

    return results, action_counts, success_counts, total_reward


def print_report(results, action_counts, success_counts, total_reward):
    print("\n===== Detailed Runs =====\n")
    for r in results:
        print(f"{r['file']}: action={r['action']}, success={r['success']}, "
              f"real_success={r['real_success']}, reward={r['reward']}")

    print("\n===== Summary =====\n")

    for action in action_counts:
        count = action_counts[action]
        success = success_counts[action]
        avg_reward = total_reward[action] / count

        print(f"Action: {action}")
        print(f"  count: {count}")
        print(f"  success_rate: {success}/{count} = {success/count:.2f}")
        print(f"  avg_reward: {avg_reward:.2f}")
        print()


def main():
    run_agent_n_times(RUN_TIMES)

    files = load_latest_runs(RUN_TIMES)

    results, action_counts, success_counts, total_reward = analyze_runs(files)

    print_report(results, action_counts, success_counts, total_reward)


if __name__ == "__main__":
    main()
