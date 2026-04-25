import torch
import torch.nn as nn


MODEL_PATH = "rl_agent/softq_model.pt"


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


def predict_action(state_summary):
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    model = SoftQNet(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = torch.tensor([state_to_vector(state_summary)], dtype=torch.float32)

    with torch.no_grad():
        q_values = model(x)[0]
        action_idx = int(torch.argmax(q_values).item())

    idx_to_action = checkpoint["idx_to_action"]
    if isinstance(idx_to_action, dict):
        # torch save/load 可能把 key 变成字符串
        action = idx_to_action.get(action_idx, idx_to_action.get(str(action_idx)))
    else:
        raise RuntimeError("idx_to_action missing or invalid")

    return action, q_values.tolist()
