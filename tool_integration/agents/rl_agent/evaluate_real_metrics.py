import argparse
import csv
import glob
import json
import statistics
from pathlib import Path

METRIC_KEYS = [
    "success",
    "execution_time",
    "total_steps",
    "tool_calls",
    "invalid_actions",
    "failed_actions",
    "cached_actions",
    "skipped_actions",
    "invalid_action_rate",
    "failed_action_rate",
    "cache_hit_rate",
    "avg_tool_time",
    "total_reward",
]


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def std(values):
    values = [v for v in values if v is not None]
    return statistics.stdev(values) if len(values) > 1 else 0.0


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


def extract_episode_metrics_from_file(path):
    """
    Supports two JSON styles:

    1. Per-run env log:
       {
         "run_id": "...",
         "metrics": {...},
         "history": [...]
       }

    2. Evaluation summary file from train_iq_online_real_optimized.py:
       {
         "success_rate": ...,
         "avg_execution_time": ...,
         "episodes": [
           {"success": 1, "execution_time": ...},
           ...
         ]
       }
    """
    data = load_json(path)
    metrics = []

    # Style 2: eval summary with episode-level metrics.
    if isinstance(data.get("episodes"), list):
        for i, ep_metrics in enumerate(data["episodes"]):
            if isinstance(ep_metrics, dict):
                item = dict(ep_metrics)
                item["_source_file"] = str(path)
                item["_episode_index"] = i
                metrics.append(item)

        if metrics:
            return metrics

    # Style 1: one env run file with a single metrics object.
    if isinstance(data.get("metrics"), dict):
        item = dict(data["metrics"])
        item["_source_file"] = str(path)
        item["_run_id"] = data.get("run_id", "")
        metrics.append(item)

    return metrics


def load_metrics_from_dir(log_dir, pattern="*.json"):
    log_dir = Path(log_dir)

    if not log_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {log_dir}")

    files = sorted(glob.glob(str(log_dir / pattern)))

    all_metrics = []

    for file in files:
        try:
            file_metrics = extract_episode_metrics_from_file(file)
            all_metrics.extend(file_metrics)
        except Exception as e:
            print(f"[WARN] Failed to read {file}: {e}")

    return all_metrics


def summarize_metrics(metrics, label):
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
    execution_times = [safe_float(m.get("execution_time"), 0.0) for m in metrics]
    tool_calls = [safe_float(m.get("tool_calls"), 0.0) for m in metrics]
    invalid_rates = [safe_float(m.get("invalid_action_rate"), 0.0) for m in metrics]
    failed_rates = [safe_float(m.get("failed_action_rate"), 0.0) for m in metrics]
    cache_rates = [safe_float(m.get("cache_hit_rate"), 0.0) for m in metrics]
    total_steps = [safe_float(m.get("total_steps"), 0.0) for m in metrics]
    rewards = [safe_float(m.get("total_reward"), None) for m in metrics]

    return {
        "label": label,
        "episodes": len(metrics),
        "success_rate": mean(successes),
        "avg_execution_time": mean(execution_times),
        "std_execution_time": std(execution_times),
        "avg_tool_calls": mean(tool_calls),
        "std_tool_calls": std(tool_calls),
        "avg_invalid_action_rate": mean(invalid_rates),
        "std_invalid_action_rate": std(invalid_rates),
        "avg_failed_action_rate": mean(failed_rates),
        "std_failed_action_rate": std(failed_rates),
        "avg_cache_hit_rate": mean(cache_rates),
        "std_cache_hit_rate": std(cache_rates),
        "avg_total_steps": mean(total_steps),
        "std_total_steps": std(total_steps),
        "avg_reward": mean(rewards),
        "std_reward": std(rewards),
    }


def percentage_change(baseline, optimized):
    """
    Positive value means optimized is larger.
    Negative value means optimized is smaller.
    """
    if baseline == 0:
        return None

    return (optimized - baseline) / baseline * 100.0


def build_comparison(baseline_summary, optimized_summary):
    fields = [
        "success_rate",
        "avg_execution_time",
        "avg_tool_calls",
        "avg_invalid_action_rate",
        "avg_failed_action_rate",
        "avg_cache_hit_rate",
        "avg_total_steps",
        "avg_reward",
    ]

    comparison = []

    for field in fields:
        b = baseline_summary.get(field, 0.0)
        o = optimized_summary.get(field, 0.0)
        pct = percentage_change(b, o)

        comparison.append(
            {
                "metric": field,
                "baseline": b,
                "optimized": o,
                "absolute_change": o - b,
                "percent_change": pct,
            }
        )

    return comparison


def print_summary(summary):
    print("=" * 80)
    print(f"SUMMARY: {summary['label']}")
    print("=" * 80)
    print(f"episodes:                  {summary['episodes']}")
    print(f"success rate:              {summary['success_rate']:.4f}")
    print(
        f"avg execution time:        {summary['avg_execution_time']:.4f} ± {summary['std_execution_time']:.4f} sec"
    )
    print(
        f"avg tool calls:            {summary['avg_tool_calls']:.4f} ± {summary['std_tool_calls']:.4f}"
    )
    print(
        f"avg invalid action rate:   {summary['avg_invalid_action_rate']:.4f} ± {summary['std_invalid_action_rate']:.4f}"
    )
    print(
        f"avg failed action rate:    {summary['avg_failed_action_rate']:.4f} ± {summary['std_failed_action_rate']:.4f}"
    )
    print(
        f"avg cache hit rate:        {summary['avg_cache_hit_rate']:.4f} ± {summary['std_cache_hit_rate']:.4f}"
    )
    print(
        f"avg total steps:           {summary['avg_total_steps']:.4f} ± {summary['std_total_steps']:.4f}"
    )
    print(
        f"avg reward:                {summary['avg_reward']:.4f} ± {summary['std_reward']:.4f}"
    )


def print_comparison(comparison):
    print("=" * 80)
    print("COMPARISON: OPTIMIZED - BASELINE")
    print("=" * 80)

    for row in comparison:
        pct = row["percent_change"]

        if pct is None:
            pct_text = "n/a"
        else:
            pct_text = f"{pct:+.2f}%"

        print(
            f"{row['metric']:<28} "
            f"baseline={row['baseline']:.4f}  "
            f"optimized={row['optimized']:.4f}  "
            f"change={row['absolute_change']:+.4f}  "
            f"pct={pct_text}"
        )


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return str(path)


def save_csv(path, baseline_summary, optimized_summary, comparison):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "section",
                "metric",
                "baseline",
                "optimized",
                "absolute_change",
                "percent_change",
            ]
        )

        for row in comparison:
            writer.writerow(
                [
                    "comparison",
                    row["metric"],
                    row["baseline"],
                    row["optimized"],
                    row["absolute_change"],
                    "" if row["percent_change"] is None else row["percent_change"],
                ]
            )

        writer.writerow([])
        writer.writerow(["summary", "metric", "baseline", "optimized", "", ""])

        for key in baseline_summary:
            if key == "label":
                continue

            writer.writerow(
                [
                    "summary",
                    key,
                    baseline_summary.get(key),
                    optimized_summary.get(key),
                    "",
                    "",
                ]
            )

    return str(path)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate real-environment execution metrics for baseline and optimized PT agents."
    )

    parser.add_argument(
        "--baseline_dir",
        type=str,
        required=True,
        help="Directory containing baseline JSON logs or eval files.",
    )

    parser.add_argument(
        "--optimized_dir",
        type=str,
        required=True,
        help="Directory containing optimized JSON logs or eval files.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="JSON file pattern to read. Default: *.json",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="tool_integration/outputs/metric_comparison",
        help="Directory for summary JSON/CSV output.",
    )

    parser.add_argument(
        "--baseline_label",
        type=str,
        default="baseline",
    )

    parser.add_argument(
        "--optimized_label",
        type=str,
        default="optimized",
    )

    args = parser.parse_args()

    baseline_metrics = load_metrics_from_dir(args.baseline_dir, args.pattern)
    optimized_metrics = load_metrics_from_dir(args.optimized_dir, args.pattern)

    baseline_summary = summarize_metrics(baseline_metrics, args.baseline_label)
    optimized_summary = summarize_metrics(optimized_metrics, args.optimized_label)

    comparison = build_comparison(baseline_summary, optimized_summary)

    print_summary(baseline_summary)
    print_summary(optimized_summary)
    print_comparison(comparison)

    payload = {
        "baseline_dir": args.baseline_dir,
        "optimized_dir": args.optimized_dir,
        "baseline_summary": baseline_summary,
        "optimized_summary": optimized_summary,
        "comparison": comparison,
        "baseline_episodes": baseline_metrics,
        "optimized_episodes": optimized_metrics,
    }

    output_dir = Path(args.output_dir)
    json_path = save_json(output_dir / "real_metrics_comparison.json", payload)
    csv_path = save_csv(
        output_dir / "real_metrics_comparison.csv",
        baseline_summary,
        optimized_summary,
        comparison,
    )

    print("=" * 80)
    print("[SAVED]")
    print(f"json: {json_path}")
    print(f"csv:  {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
