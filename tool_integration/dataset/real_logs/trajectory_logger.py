import json
from pathlib import Path


def save_trajectory(data, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    trajectory = {
        "trajectory_id": "traj_001",
        "target": "10.11.202.189",
        "steps": [
            {
                "step": 0,
                "action": "scan_nmap",
                "result": {
                    "open_ports": [21, 22, 23, 25, 53, 80, 111, 139, 445, 512, 513, 514, 1099, 1524, 2049, 2121, 3306, 5432, 5900, 6000, 6667, 8009, 8180],
                    "key_services": [
                        "vsftpd 2.3.4",
                        "Samba 3.0.20-Debian",
                        "bindshell 1524",
                        "Tomcat 5.5"
                    ]
                },
                "done": False
            },
            {
                "step": 1,
                "action": "connect_1524_bindshell",
                "result": {
                    "shell_obtained": True,
                    "user": "root"
                },
                "reward": 10,
                "done": True
            }
        ]
    }

    save_trajectory(trajectory, "real_logs/trajectories/traj_001_from_py.json")
    print("Saved trajectory to real_logs/trajectories/traj_001_from_py.json")
