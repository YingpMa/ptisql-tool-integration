import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


TRAJ_DIR = Path("real_logs/trajectories")
MODEL_PATH = Path("rl_agent/softq_model.pt")

ACTION_TO_IDX = {
    "connect_1524_bindshell": 0,
    "exploit_vsftpd": 1,
    "no_exploit": 2,
}

IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}


def state_to_vector(state):
    return [
        float(state.get("num_open_ports", 0)),
        float(state.get("has_ftp", False)),
        float(state.get("has_ssh", False)),
        float(state.get("has_telnet", False)),
        float(state.get("has_http", False)),
        float(state.get("has_mysql", False)),
        float(state.get("has_postgresql", False)),
        float(state.get("has_bindshell_1524", False)),
        float(state.get("has_samba", False)),
        float(state.get("has_tomcat", False)),
        float(state.get("has_vsftpd_234", False)),
    ]


def load_dataset():
    samples = []

    for path in sorted(TRAJ_DIR.glob("batch_traj_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            traj = json.load(f)

        try:
            state = traj["steps"][0]["result"]["state_summary"]
            action = traj["steps"][1]["result"]["chosen_action"]
            reward = float(traj["steps"][2]["reward"])

            samples.append(
                (
                    state_to_vector(state),
                    ACTION_TO_IDX[action],
                    reward,
                )
            )
        except Exception as e:
            print(f"[!] Skip {path.name}: {e}")

    return samples


class SoftQNet(nn.Module):
    def __init__(self, state_dim=11, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    samples = load_dataset()
    if not samples:
        raise RuntimeError("No training samples found.")

    X = torch.tensor([s[0] for s in samples], dtype=torch.float32)
    A = torch.tensor([s[1] for s in samples], dtype=torch.long)
    R = torch.tensor([s[2] for s in samples], dtype=torch.float32)

    model = SoftQNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 对于你现在这种单步决策，直接拟合 Q(s,a) ≈ reward
    for epoch in range(300):
        q_values = model(X)                     # [N, 3]
        q_selected = q_values.gather(1, A.unsqueeze(1)).squeeze(1)
        loss = ((q_selected - R) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"epoch={epoch}, loss={loss.item():.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dim": 11,
            "action_dim": 3,
            "model_state_dict": model.state_dict(),
            "action_to_idx": ACTION_TO_IDX,
            "idx_to_action": IDX_TO_ACTION,
        },
        MODEL_PATH,
    )

    print(f"[✓] Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
	
