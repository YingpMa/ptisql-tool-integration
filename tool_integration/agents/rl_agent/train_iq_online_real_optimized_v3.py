"""
Optimized v3 training/evaluation entry point.

Place this file at:
  tool_integration/agents/rl_agent/train_iq_online_real_optimized_v3.py

It reuses the original optimized training script and only switches the
environment to pt_env_optimized_v3.RealPTEnv.

v3 is an aggressive speed-oriented execution variant:
- scan_basic still uses Nmap for bounded port discovery
- scan_service uses direct service inference / banner grabbing instead of
  full Nmap service-version detection

Important:
This file should import train_iq_online_real_optimized as base_train.
Do not import train_iq_online_real_optimized_v2 here, because v2 is itself
only a wrapper and does not expose main() directly.
"""

import tool_integration.agents.rl_agent.train_iq_online_real_optimized as base_train


def make_env(args):
    from tool_integration.agents.rl_agent.pt_env_optimized_v3 import RealPTEnv

    return RealPTEnv(
        target_ip=args.target_ip,
        use_metasploit=args.use_metasploit,
    )


# Monkey-patch the environment factory used by the original optimized script.
base_train.make_env = make_env


if __name__ == "__main__":
    base_train.main()
