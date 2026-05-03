"""
Optimized v2 training/evaluation entry point.

Place this file at:
  tool_integration/agents/rl_agent/train_iq_online_real_optimized_v2.py

It reuses the optimized v1 training script and only switches the environment to
pt_env_optimized_v2.RealPTEnv. This keeps the agent/training code identical
between optimized v1 and optimized v2, so the comparison is cleaner.
"""

import os

import tool_integration.agents.rl_agent.train_iq_online_real_optimized as base_train


def make_env(args):
    from tool_integration.agents.rl_agent.pt_env_optimized_v2 import RealPTEnv

    return RealPTEnv(
        target_ip=args.target_ip,
        use_metasploit=args.use_metasploit,
    )


# Monkey-patch the environment factory used by the original optimized script.
base_train.make_env = make_env


if __name__ == "__main__":
    base_train.main()
