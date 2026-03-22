import json
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ACTION_CLASSES = [
    "SubnetScan",
    "OSScan",
    "ServiceScan",
    "ProcessScan",
    "HTTP-EXP",
    "SSH-EXP",
    "FTP-EXP",
    "Tomcat-PE",
]
ACTION_TO_IDX = {name: i for i, name in enumerate(ACTION_CLASSES)}


def normalize_action(action_name: str):
    action_name = action_name.strip()
    if action_name in ACTION_TO_IDX:
        return action_name
    if action_name.lower() == "daclsvc-pe":
        return None
    return None


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_samples(data_dir: Path):
    xs = []
    ys = []
    skipped_bad_state = 0
    skipped_unknown_action = 0

    for json_file in sorted(data_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for step in data.get("steps", []):
            action = normalize_action(step.get("action", ""))
            if action is None:
                skipped_unknown_action += 1
                continue

            state = step.get("state")
            if not isinstance(state, list) or len(state) != 40:
                skipped_bad_state += 1
                continue

            xs.append(state)
            ys.append(ACTION_TO_IDX[action])

    return xs, ys, skipped_bad_state, skipped_unknown_action


def split_dataset(xs, ys, val_ratio=0.2, seed=0):
    indices = list(range(len(xs)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = max(1, int(len(indices) * val_ratio)) if len(indices) > 1 else len(indices)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    if len(train_idx) == 0 and len(val_idx) > 0:
        train_idx = val_idx[:1]
        val_idx = val_idx[1:]

    def gather(idxs):
        x = torch.tensor([xs[i] for i in idxs], dtype=torch.float32)
        y = torch.tensor([ys[i] for i in idxs], dtype=torch.long)
        return x, y

    return gather(train_idx), gather(val_idx)


def accuracy(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            total += yb.numel()
            correct += (preds == yb).sum().item()
            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(yb.cpu().tolist())
    acc = correct / total if total > 0 else 0.0
    return acc, preds_all, labels_all


def confusion_matrix(preds, labels, num_classes):
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for y, p in zip(labels, preds):
        matrix[y][p] += 1
    return matrix


def print_confusion_matrix(matrix):
    header = ["true\\pred"] + ACTION_CLASSES
    print("\t".join(header))
    for i, row in enumerate(matrix):
        print("\t".join([ACTION_CLASSES[i]] + [str(v) for v in row]))


def main():
    data_dir = Path("real_engagements")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xs, ys, skipped_bad_state, skipped_unknown_action = load_samples(data_dir)
    if len(xs) == 0:
        print("No valid samples found.")
        return

    label_counts = Counter(ys)
    print("dataset_size", len(xs))
    print("skipped_bad_state", skipped_bad_state)
    print("skipped_unknown_action", skipped_unknown_action)
    print("label_distribution")
    for idx, count in sorted(label_counts.items()):
        print(ACTION_CLASSES[idx], count)

    (x_train, y_train), (x_val, y_val) = split_dataset(xs, ys, val_ratio=0.2, seed=0)

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=min(16, max(1, len(train_ds))), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(16, max(1, len(val_ds))), shuffle=False)

    model = MLP(input_dim=40, output_dim=len(ACTION_CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 30
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        train_acc, _, _ = accuracy(model, train_loader, device)
        val_acc, _, _ = accuracy(model, val_loader, device)
        print(f"epoch {epoch:02d} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    train_acc, train_preds, train_labels = accuracy(model, train_loader, device)
    val_acc, val_preds, val_labels = accuracy(model, val_loader, device)

    print("final_train_accuracy", train_acc)
    print("final_validation_accuracy", val_acc)

    if len(val_labels) > 0:
        print("validation_confusion_matrix")
        matrix = confusion_matrix(val_preds, val_labels, len(ACTION_CLASSES))
        print_confusion_matrix(matrix)


if __name__ == "__main__":
    main()