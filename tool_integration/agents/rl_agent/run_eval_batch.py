import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path


def run_one(model_type, model_path, replay_path, seed, episodes, max_steps, out_dir):
  log_path = out_dir / f"{model_type}_seed{seed}.log"

  cmd = [
    "python", "-m", "rl_agent.eval_pt_agents",
    "--model_path", model_path,
    "--model_type", model_type,
    "--replay_path", replay_path,
    "--episodes", str(episodes),
    "--max_steps", str(max_steps),
    "--seed", str(seed),
  ]

  print("[RUN]", " ".join(cmd))

  with open(log_path, "w", encoding="utf-8") as f:
    p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

  if p.returncode != 0:
    print(f"[ERROR] {model_type} seed={seed}, see {log_path}")
    return None

  text = log_path.read_text(encoding="utf-8")

  def grab(name):
    m = re.search(rf"{name}=([-+]?\d+\.\d+)", text)
    return float(m.group(1)) if m else None

  return {
    "model": model_type,
    "seed": seed,
    "episodes": episodes,
    "max_steps": max_steps,
    "avg_reward": grab("avg_reward"),
    "std_reward": grab("std_reward"),
    "goal_reached_rate": grab("goal_reached_rate"),
    "honeypot_rate": grab("honeypot_rate"),
    "avg_steps": grab("avg_steps"),
    "log_path": str(log_path),
  }


def summarize(rows):
  metrics = ["avg_reward", "goal_reached_rate", "honeypot_rate", "avg_steps"]
  summary = {}

  for model in sorted(set(r["model"] for r in rows)):
    model_rows = [r for r in rows if r["model"] == model]
    summary[model] = {}

    for m in metrics:
      vals = [r[m] for r in model_rows if r[m] is not None]
      if vals:
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        summary[model][m] = {
          "mean": mean,
          "std": var ** 0.5,
          "n": len(vals),
        }

  return summary


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--iq_model", default="outputs/iq_replay/best_iq_replay.pt")
  parser.add_argument("--bc_model", default="outputs/bc_replay/best_bc_replay.pt")
  parser.add_argument("--replay_path", default="outputs/replay_iq_650.json")
  parser.add_argument("--seeds", default="1,2,3,4,5")
  parser.add_argument("--episodes", type=int, default=20)
  parser.add_argument("--max_steps", type=int, default=10)
  parser.add_argument("--out_dir", default="outputs/eval_batch")
  args = parser.parse_args()

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
  rows = []

  for seed in seeds:
    rows.append(run_one("iq", args.iq_model, args.replay_path, seed, args.episodes, args.max_steps, out_dir))
    rows.append(run_one("bc", args.bc_model, args.replay_path, seed, args.episodes, args.max_steps, out_dir))

  rows = [r for r in rows if r is not None]

  csv_path = out_dir / "eval_results.csv"
  with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  summary = summarize(rows)
  summary_path = out_dir / "summary.json"
  summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

  print("\n[DONE]")
  print(f"Raw logs: {out_dir}")
  print(f"CSV: {csv_path}")
  print(f"Summary: {summary_path}")
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
