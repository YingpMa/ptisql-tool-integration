import numpy as np
import torch
from types import SimpleNamespace

from real_env_demo import RealPentestEnv
from agent.softq import SoftQ


def build_dummy_args(obs_dim, action_dim):
    critic_cfg = {
        "_target_": "agent.softq_models.SimpleQ2Network",
        "obs_dim": obs_dim,
        "action_dim": action_dim,
    }

    agent_cfg = SimpleNamespace(
        name="softq",
        critic_tau=0.1,
        critic_target_update_frequency=4,
        init_temp=0.01,
        critic_cfg=critic_cfg,
        critic_lr=1e-4,
        critic_betas=(0.9, 0.999),
    )

    args = SimpleNamespace(
        gamma=0.99,
        device="cpu",
        agent=agent_cfg,
    )
    return args


def load_iq_policy(path, obs_dim, action_dim):
    args = build_dummy_args(obs_dim, action_dim)
    agent = SoftQ(obs_dim, action_dim, batch_size=32, args=args)
    agent.load(path)
    return agent


def iq_policy(agent, obs):
    action = agent.choose_action(obs, sample=False)
    return int(action)


def main():
    model_path = "outputs/2026-03-22/07-26-00/results_best/softq_iq_nasim:SmallHoneypotPO-v0"

    env = RealPentestEnv()
    obs, info = env.reset()
    print("reset:", obs)

    print("Trying to load NASim IQ model...")
    agent = load_iq_policy(model_path, obs_dim=4, action_dim=2)
    print("Model loaded successfully.")

    for t in range(3):
        action = iq_policy(agent, obs)
        print(f"step {t}, action = {action}")

        obs, reward, terminated, truncated, info = env.step(action)

        print("obs:", obs)
        print("reward:", reward)
        print("terminated:", terminated)
        print("truncated:", truncated)

        if "raw_output" in info:
            print("raw_output:")
            print(info["raw_output"])

        if terminated or truncated:
            print("episode finished")
            break


if __name__ == "__main__":
    main()
