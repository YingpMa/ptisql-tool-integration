"""
Compare baseline, optimized v1, and optimized v2 execution metrics.

Place this file at:
  tool_integration/agents/rl_agent/evaluate_real_metrics_v2.py

Example:
python -m tool_integration.agents.rl_agent.evaluate_real_metrics_v2 \
  --baseline_dir tool_integration/dataset/real_logs/rl_env_runs \
  --optimized_v1_dir tool_integration/dataset/real_logs/rl_env_runs_optimized \
  --optimized_v2_dir tool_integration/dataset/real_logs/rl_env_runs_optimized_v2 \
  --output_dir tool_integration/outputs/metric_comparison_v2
"""

import argparse
import csv
import glob
import json
import statistics
from pathlib import Path

SUMMARY_KEYS = [
    "episodes",
    "success_rate",
    "avg_execution_time",
    "std_execution_time",
    "avg_tool_calls",
    "std_tool_calls",
    "avg_invalid_action_rate",
    "std_invalid_action_rate",
    "avg_failed_action_rate",
    "std_failed_action_rate",
    "avg_cache_hit_rate",
    "std_cache_hit_rate",
    "avg_total_steps",
    "std_total_steps",
    "avg_reward",
    "std_reward",
]

COMPARISON_KEYS = [
    "success_rate",
    "avg_execution_time",
    "avg_tool_calls",
    "avg_invalid_action_rate",
    "avg_failed_action_rate",
    "avg_cache_hit_rate",
    "avg_total_steps",
    "avg_reward",
]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def safe_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def reward_from_history(data):
    history = data.get("history", [])
    total = 0.0
    seen = False
    if isinstance(history, list):
        for step in history:
            if isinstance(step, dict) and "reward" in step:
                total += safe_float(step.get("reward"), 0.0)
                seen = True
    return total if seen else None


def extract_episode_metrics_from_file(path):
    data = load_json(path)
    rows = []

    # Evaluation summary style: {"episodes": [{...}, ...]}
    if isinstance(data.get("episodes"), list):
        for idx, item in enumerate(data["episodes"]):
            if isinstance(item, dict):
                m = dict(item)
                m.setdefault("total_reward", item.get("total_reward"))
                m["_source_file"] = str(path)
                m["_episode_index"] = idx
                rows.append(m)
        if rows:
            return rows

    # Per-run env log style: {"metrics": {...}, "history": [...]}
    if isinstance(data.get("metrics"), dict):
        m = dict(data["metrics"])
        if "total_reward" not in m or m.get("total_reward") is None:
            hist_reward = reward_from_history(data)
            if hist_reward is not None:
                m["total_reward"] = hist_reward
        m["_source_file"] = str(path)
        m["_run_id"] = data.get("run_id", "")
        rows.append(m)

    return rows


def load_metrics_from_dir(log_dir, pattern="*.json"):
    log_dir = Path(log_dir)
    if not log_dir.exists():
        print(f"[WARN] Directory does not exist: {log_dir}")
        return []

    files = sorted(glob.glob(str(log_dir / pattern)))
    rows = []
    for file in files:
        try:
            rows.extend(extract_episode_metrics_from_file(file))
        except Exception as e:
            print(f"[WARN] Failed to read {file}: {e}")
    return rows


def summarize(metrics, label):
    if not metrics:
        return {
            "label": label,
            "episodes": 0,
            "success_rate": 0.0,
            "avg_execution_time": 0.0,
            "std_execution_time": 0.0,
            "avg_tool_calls": 0.0,
            "std_tool_calls": 0.0,
            "avg_invalid_action_rate": 0.0,
            "std_invalid_action_rate": 0.0,
            "avg_failed_action_rate": 0.0,
            "std_failed_action_rate": 0.0,
            "avg_cache_hit_rate": 0.0,
            "std_cache_hit_rate": 0.0,
            "avg_total_steps": 0.0,
            "std_total_steps": 0.0,
            "avg_reward": 0.0,
            "std_reward": 0.0,
        }

    successes = [safe_float(m.get("success"), 0.0) for m in metrics]
    times = [safe_float(m.get("execution_time"), 0.0) for m in metrics]
    calls = [safe_float(m.get("tool_calls"), 0.0) for m in metrics]
    invalid_rates = [safe_float(m.get("invalid_action_rate"), 0.0) for m in metrics]
    failed_rates = [safe_float(m.get("failed_action_rate"), 0.0) for m in metrics]
    cache_rates = [safe_float(m.get("cache_hit_rate"), 0.0) for m in metrics]
    steps = [safe_float(m.get("total_steps"), 0.0) for m in metrics]
    rewards = [safe_float(m.get("total_reward"), None) for m in metrics]

    return {
        "label": label,
        "episodes": len(metrics),
        "success_rate": mean(successes),
        "avg_execution_time": mean(times),
        "std_execution_time": std(times),
        "avg_tool_calls": mean(calls),
        "std_tool_calls": std(calls),
        "avg_invalid_action_rate": mean(invalid_rates),
        "std_invalid_action_rate": std(invalid_rates),
        "avg_failed_action_rate": mean(failed_rates),
        "std_failed_action_rate": std(failed_rates),
        "avg_cache_hit_rate": mean(cache_rates),
        "std_cache_hit_rate": std(cache_rates),
        "avg_total_steps": mean(steps),
        "std_total_steps": std(steps),
        "avg_reward": mean(rewards),
        "std_reward": std(rewards),
    }


def pct_change(base, other):
    if base == 0:
        return None
    return (other - base) / base * 100.0


def make_pairwise_comparison(a, b, a_name, b_name):
    rows = []
    for key in COMPARISON_KEYS:
        av = a.get(key, 0.0)
        bv = b.get(key, 0.0)
        rows.append(
            {
                "comparison": f"{b_name} - {a_name}",
                "metric": key,
                a_name: av,
                b_name: bv,
                "absolute_change": bv - av,
                "percent_change": pct_change(av, bv),
            }
        )
    return rows


def print_summary(s):
    print("=" * 90)
    print(f"SUMMARY: {s['label']}")
    print("=" * 90)
    print(f"episodes:                {s['episodes']}")
    print(f"success rate:            {s['success_rate']:.4f}")
    print(
        f"avg execution time:      {s['avg_execution_time']:.4f} ± {s['std_execution_time']:.4f} sec"
    )
    print(
        f"avg tool calls:          {s['avg_tool_calls']:.4f} ± {s['std_tool_calls']:.4f}"
    )
    print(
        f"avg invalid action rate: {s['avg_invalid_action_rate']:.4f} ± {s['std_invalid_action_rate']:.4f}"
    )
    print(
        f"avg failed action rate:  {s['avg_failed_action_rate']:.4f} ± {s['std_failed_action_rate']:.4f}"
    )
    print(
        f"avg cache hit rate:      {s['avg_cache_hit_rate']:.4f} ± {s['std_cache_hit_rate']:.4f}"
    )
    print(
        f"avg total steps:         {s['avg_total_steps']:.4f} ± {s['std_total_steps']:.4f}"
    )
    print(f"avg reward:              {s['avg_reward']:.4f} ± {s['std_reward']:.4f}")


def save_outputs(output_dir, payload, summaries, comparisons):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "real_metrics_comparison_v2.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "real_metrics_comparison_v2.csv"
    labels = [s["label"] for s in summaries]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "metric"] + labels)
        for key in SUMMARY_KEYS:
            writer.writerow(["summary", key] + [s.get(key, "") for s in summaries])

        writer.writerow([])
        writer.writerow(
            [
                "section",
                "comparison",
                "metric",
                "from",
                "to",
                "absolute_change",
                "percent_change",
            ]
        )
        for row in comparisons:
            comp = row["comparison"]
            metric = row["metric"]
            parts = comp.split(" - ")
            to_name = parts[0]
            from_name = parts[1]
            writer.writerow(
                [
                    "comparison",
                    comp,
                    metric,
                    from_name,
                    to_name,
                    row["absolute_change"],
                    "" if row["percent_change"] is None else row["percent_change"],
                ]
            )

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline, optimized v1, and optimized v2 metrics."
    )
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--optimized_v1_dir", required=True)
    parser.add_argument("--optimized_v2_dir", required=True)
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument(
        "--output_dir", default="tool_integration/outputs/metric_comparison_v2"
    )
    args = parser.parse_args()

    baseline_metrics = load_metrics_from_dir(args.baseline_dir, args.pattern)
    v1_metrics = load_metrics_from_dir(args.optimized_v1_dir, args.pattern)
    v2_metrics = load_metrics_from_dir(args.optimized_v2_dir, args.pattern)

    baseline = summarize(baseline_metrics, "baseline")
    v1 = summarize(v1_metrics, "optimized_v1")
    v2 = summarize(v2_metrics, "optimized_v2")

    summaries = [baseline, v1, v2]
    for s in summaries:
        print_summary(s)

    comparisons = []
    comparisons.extend(
        make_pairwise_comparison(baseline, v1, "baseline", "optimized_v1")
    )
    comparisons.extend(
        make_pairwise_comparison(baseline, v2, "baseline", "optimized_v2")
    )
    comparisons.extend(make_pairwise_comparison(v1, v2, "optimized_v1", "optimized_v2"))

    print("=" * 90)
    print("PAIRWISE COMPARISONS")
    print("=" * 90)
    for row in comparisons:
        pct = row["percent_change"]
        pct_text = "n/a" if pct is None else f"{pct:+.2f}%"
        print(
            f"{row['comparison']:<28} {row['metric']:<28} "
            f"change={row['absolute_change']:+.4f} pct={pct_text}"
        )

    payload = {
        "dirs": {
            "baseline": args.baseline_dir,
            "optimized_v1": args.optimized_v1_dir,
            "optimized_v2": args.optimized_v2_dir,
        },
        "summaries": {
            "baseline": baseline,
            "optimized_v1": v1,
            "optimized_v2": v2,
        },
        "comparisons": comparisons,
        "episodes": {
            "baseline": baseline_metrics,
            "optimized_v1": v1_metrics,
            "optimized_v2": v2_metrics,
        },
    }

    json_path, csv_path = save_outputs(args.output_dir, payload, summaries, comparisons)
    print("=" * 90)
    print("[SAVED]")
    print(f"json: {json_path}")
    print(f"csv:  {csv_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
