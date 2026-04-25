import json
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


TRAJ_DIR = Path("real_logs/trajectories")


def load_trajectories():
    data = []

    for path in sorted(TRAJ_DIR.glob("batch_traj_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            traj = json.load(f)

        try:
            step0 = traj["steps"][0]
            step1 = traj["steps"][1]

            state = step0["result"]["state_summary"]
            action = step1["result"]["chosen_action"]

            data.append((state, action, path.name))
        except Exception as e:
            print(f"[!] Skip {path.name}: {e}")

    return data


def state_to_vector(state):
    """
    把 state_summary 转成固定长度向量
    """
    return [
        int(state.get("num_open_ports", 0)),
        int(state.get("has_ftp", False)),
        int(state.get("has_ssh", False)),
        int(state.get("has_telnet", False)),
        int(state.get("has_http", False)),
        int(state.get("has_mysql", False)),
        int(state.get("has_postgresql", False)),
        int(state.get("has_bindshell_1524", False)),
        int(state.get("has_samba", False)),
        int(state.get("has_tomcat", False)),
        int(state.get("has_vsftpd_234", False)),
    ]


def main():
    dataset = load_trajectories()

    if len(dataset) < 4:
        print("[!] Not enough data. Need at least a few trajectories.")
        return

    X = []
    y = []
    names = []

    for state, action, name in dataset:
        X.append(state_to_vector(state))
        y.append(action)
        names.append(name)

    print(f"[*] Loaded {len(X)} samples")

    # 看一下标签分布
    label_count = {}
    for label in y:
        label_count[label] = label_count.get(label, 0) + 1
    print("[*] Label distribution:", label_count)

    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, names, test_size=0.3, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n[*] Classification report:")
    print(classification_report(y_test, y_pred))

    print("[*] Test predictions:")
    for name, true_label, pred_label in zip(names_test, y_test, y_pred):
        print(f"  {name}: true={true_label}, pred={pred_label}")

    # 保存一个简单模型结果说明
    model_info = {
        "num_samples": len(X),
        "label_distribution": label_count,
        "feature_order": [
            "num_open_ports",
            "has_ftp",
            "has_ssh",
            "has_telnet",
            "has_http",
            "has_mysql",
            "has_postgresql",
            "has_bindshell_1524",
            "has_samba",
            "has_tomcat",
            "has_vsftpd_234",
        ]
    }

    with open("real_logs/bc_model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    print("\n[✓] Saved model info to real_logs/bc_model_info.json")


if __name__ == "__main__":
    main()
