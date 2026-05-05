import argparse
import csv
import glob
import json
import statistics
from pathlib import Path

FIELDS = [
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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def total_reward_from_history(data):
    return sum(
        float(step.get("reward", 0.0))
        for step in data.get("history", [])
        if isinstance(step, dict)
    )


def extract_metrics(path):
    data = load_json(path)
    out = []

    if isinstance(data.get("episodes"), list):
        for ep in data["episodes"]:
            if isinstance(ep, dict):
                m = dict(ep)
                if "total_reward" not in m:
                    m["total_reward"] = m.get("avg_reward", 0.0)
                out.append(m)
        return out

    if isinstance(data.get("metrics"), dict):
        m = dict(data["metrics"])
        if "total_reward" not in m:
            m["total_reward"] = total_reward_from_history(data)
        out.append(m)

    return out


def load_dir(directory):
    rows = []
    directory = Path(directory)

    if not directory.exists():
        print(f"[WARN] directory does not exist: {directory}")
        return rows

    for path in sorted(glob.glob(str(directory / "*.json"))):
        try:
            rows.extend(extract_metrics(path))
        except Exception as e:
            print(f"[WARN] failed to read {path}: {e}")

    return rows


def summarize(label, rows):
    success = [float(r.get("success", 0.0)) for r in rows]
    times = [float(r.get("execution_time", 0.0)) for r in rows]
    calls = [float(r.get("tool_calls", 0.0)) for r in rows]
    invalid = [float(r.get("invalid_action_rate", 0.0)) for r in rows]
    failed = [float(r.get("failed_action_rate", 0.0)) for r in rows]
    cache = [float(r.get("cache_hit_rate", 0.0)) for r in rows]
    steps = [float(r.get("total_steps", 0.0)) for r in rows]
    reward = [float(r.get("total_reward", 0.0)) for r in rows]

    return {
        "label": label,
        "episodes": len(rows),
        "success_rate": mean(success),
        "avg_execution_time": mean(times),
        "std_execution_time": std(times),
        "avg_tool_calls": mean(calls),
        "std_tool_calls": std(calls),
        "avg_invalid_action_rate": mean(invalid),
        "std_invalid_action_rate": std(invalid),
        "avg_failed_action_rate": mean(failed),
        "std_failed_action_rate": std(failed),
        "avg_cache_hit_rate": mean(cache),
        "std_cache_hit_rate": std(cache),
        "avg_total_steps": mean(steps),
        "std_total_steps": std(steps),
        "avg_reward": mean(reward),
        "std_reward": std(reward),
    }


def pct_change(a, b):
    if a == 0:
        return None
    return (b - a) / a * 100.0


def pairwise(name, src, dst):
    rows = []
    for field in FIELDS:
        a = src[field]
        b = dst[field]
        rows.append(
            {
                "comparison": name,
                "metric": field,
                "from": src["label"],
                "to": dst["label"],
                "absolute_change": b - a,
                "percent_change": pct_change(a, b),
            }
        )
    return rows


def print_summary(summaries, comparisons):
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)

    for s in summaries:
        print(
            f"{s['label']:<14} episodes={s['episodes']:<5} "
            f"success={s['success_rate']:.4f} "
            f"time={s['avg_execution_time']:.2f}±{s['std_execution_time']:.2f}s "
            f"calls={s['avg_tool_calls']:.2f} "
            f"invalid={s['avg_invalid_action_rate']:.4f} "
            f"failed={s['avg_failed_action_rate']:.4f} "
            f"reward={s['avg_reward']:.3f}"
        )

    print("=" * 110)
    print("PAIRWISE")
    print("=" * 110)

    for c in comparisons:
        pct = "n/a" if c["percent_change"] is None else f"{c['percent_change']:+.2f}%"
        print(
            f"{c['comparison']:<32} {c['metric']:<28} "
            f"change={c['absolute_change']:+.4f} pct={pct}"
        )


def write_csv(path, summaries, comparisons):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)

        w.writerow(["section", "metric"] + [s["label"] for s in summaries])

        keys = [
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

        for key in keys:
            w.writerow(["summary", key] + [s.get(key, "") for s in summaries])

        w.writerow([])
        w.writerow(
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

        for c in comparisons:
            w.writerow(
                [
                    "comparison",
                    c["comparison"],
                    c["metric"],
                    c["from"],
                    c["to"],
                    c["absolute_change"],
                    "" if c["percent_change"] is None else c["percent_change"],
                ]
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--optimized_v1_dir", required=True)
    parser.add_argument("--optimized_v2_dir", required=True)
    parser.add_argument("--optimized_v25_dir", required=True)
    parser.add_argument("--optimized_v3_dir", default="")
    parser.add_argument(
        "--output_dir", default="tool_integration/outputs/metric_comparison_v25"
    )

    args = parser.parse_args()

    summaries = [
        summarize("baseline", load_dir(args.baseline_dir)),
        summarize("optimized_v1", load_dir(args.optimized_v1_dir)),
        summarize("optimized_v2", load_dir(args.optimized_v2_dir)),
        summarize("optimized_v25", load_dir(args.optimized_v25_dir)),
    ]

    if args.optimized_v3_dir:
        summaries.append(summarize("optimized_v3", load_dir(args.optimized_v3_dir)))

    by_label = {s["label"]: s for s in summaries}

    comparisons = []
    comparisons += pairwise(
        "optimized_v1 - baseline", by_label["baseline"], by_label["optimized_v1"]
    )
    comparisons += pairwise(
        "optimized_v2 - baseline", by_label["baseline"], by_label["optimized_v2"]
    )
    comparisons += pairwise(
        "optimized_v25 - baseline", by_label["baseline"], by_label["optimized_v25"]
    )
    comparisons += pairwise(
        "optimized_v25 - optimized_v2",
        by_label["optimized_v2"],
        by_label["optimized_v25"],
    )

    if "optimized_v3" in by_label:
        comparisons += pairwise(
            "optimized_v3 - baseline", by_label["baseline"], by_label["optimized_v3"]
        )
        comparisons += pairwise(
            "optimized_v3 - optimized_v25",
            by_label["optimized_v25"],
            by_label["optimized_v3"],
        )

    print_summary(summaries, comparisons)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "summaries": summaries,
        "comparisons": comparisons,
    }

    json_path = output_dir / "real_metrics_comparison_v25.json"
    csv_path = output_dir / "real_metrics_comparison_v25.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    write_csv(csv_path, summaries, comparisons)

    print("=" * 110)
    print(f"[SAVED] {json_path}")
    print(f"[SAVED] {csv_path}")


if __name__ == "__main__":
    main()
