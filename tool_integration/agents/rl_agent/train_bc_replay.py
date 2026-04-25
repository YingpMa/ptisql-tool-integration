import argparse
import json
import os
import random
from collections import Counter, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BCPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_replay(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    trajectories = data["trajectories"] if isinstance(data, dict) else data

    actions = sorted({step["action"] for traj in trajectories for step in traj})
    action_to_id = {a: i for i, a in enumerate(actions)}
    id_to_action = {i: a for a, i in action_to_id.items()}

    transitions = []
    for traj in trajectories:
        for step in traj:
            transitions.append(
                (
                    np.array(step["state"], dtype=np.float32),
                    action_to_id[step["action"]],
                    step["action"],
                )
            )

    return trajectories, transitions, action_to_id, id_to_action


def split_transitions(transitions, train_ratio):
    random.shuffle(transitions)
    split = int(len(transitions) * train_ratio)
    return transitions[:split], transitions[split:]


def make_batch(transitions, batch_size, device):
    batch = random.sample(transitions, batch_size)
    states, action_ids, _ = zip(*batch)

    return (
        torch.tensor(np.array(states), dtype=torch.float32, device=device),
        torch.tensor(action_ids, dtype=torch.long, device=device),
    )


@torch.no_grad()
def evaluate(model, transitions, id_to_action, device):
    model.eval()

    states = torch.tensor(
        np.array([x[0] for x in transitions]),
        dtype=torch.float32,
        device=device,
    )
    labels = torch.tensor(
        [x[1] for x in transitions],
        dtype=torch.long,
        device=device,
    )

    logits = model(states)
    preds = torch.argmax(logits, dim=1)

    acc = (preds == labels).float().mean().item()

    pred_counter = Counter(id_to_action[int(x)] for x in preds.cpu().numpy())
    expert_counter = Counter(x[2] for x in transitions)

    model.train()

    return acc, pred_counter, expert_counter


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--replay_path", type=str, default="outputs/replay_iq_650.json")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="outputs/bc_replay")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    trajectories, transitions, action_to_id, id_to_action = load_replay(args.replay_path)
    train_data, eval_data = split_transitions(transitions, args.train_ratio)

    state_dim = len(transitions[0][0])
    action_dim = len(action_to_id)

    print("=" * 60)
    print(f"[DATA] replay_path = {args.replay_path}")
    print(f"[DATA] trajectories = {len(trajectories)}")
    print(f"[DATA] transitions = {len(transitions)}")
    print(f"[DATA] train transitions = {len(train_data)}")
    print(f"[DATA] eval transitions = {len(eval_data)}")
    print(f"[DATA] state_dim = {state_dim}")
    print(f"[DATA] action_dim = {action_dim}")
    print(f"[DATA] actions = {list(action_to_id.keys())}")
    print(f"[DEVICE] {device}")
    print("=" * 60)

    model = BCPolicy(state_dim, action_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)

    recent_losses = deque(maxlen=100)
    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        states, labels = make_batch(train_data, args.batch_size, device)

        logits = model(states)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recent_losses.append(loss.item())

        if epoch % args.log_interval == 0:
            print(f"[Epoch {epoch}] loss={np.mean(recent_losses):.6f}")

        if epoch % args.eval_interval == 0:
            acc, pred_counter, expert_counter = evaluate(model, eval_data, id_to_action, device)

            print("-" * 60)
            print(f"[EVAL {epoch}] matched_ratio={acc:.4f}")
            print(f"[EVAL {epoch}] pred_actions={pred_counter}")
            print(f"[EVAL {epoch}] expert_actions={expert_counter}")
            print("-" * 60)

            if acc > best_acc:
                best_acc = acc
                save_path = os.path.join(args.save_dir, "best_bc_replay.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "state_dim": state_dim,
                        "action_dim": action_dim,
                        "hidden_dim": args.hidden_dim,
                        "action_to_id": action_to_id,
                        "id_to_action": id_to_action,
                        "best_acc": best_acc,
                        "args": vars(args),
                    },
                    save_path,
                )
                print(f"[SAVE] best model saved: {save_path}")

    last_path = os.path.join(args.save_dir, "last_bc_replay.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": args.hidden_dim,
            "action_to_id": action_to_id,
            "id_to_action": id_to_action,
            "args": vars(args),
        },
        last_path,
    )

    print("=" * 60)
    print("[DONE] BC training finished.")
    print(f"[DONE] best_matched = {best_acc:.4f}")
    print(f"[DONE] last model = {last_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
