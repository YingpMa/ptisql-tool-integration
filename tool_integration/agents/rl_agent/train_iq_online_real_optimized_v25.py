"""
Optimized v2.5 training/evaluation entry point.

Place this file at:
  tool_integration/agents/rl_agent/train_iq_online_real_optimized_v25.py

It reuses the optimized v1 training script and switches:
1. environment -> pt_env_optimized_v25.RealPTEnv
2. action mask -> fast-path bindshell after basic scan

This keeps the agent/training code identical to v1/v2 except for the execution
environment and valid-action mask.
"""

import tool_integration.agents.rl_agent.train_iq_online_real_optimized as base_train


def make_env(args):
    from tool_integration.agents.rl_agent.pt_env_optimized_v25 import RealPTEnv

    return RealPTEnv(
        target_ip=args.target_ip,
        use_metasploit=args.use_metasploit,
    )


def get_valid_action_names(state):
    """
    v2.5 action mask.

    Difference from v2:
    - v2 usually forces:
        scan_basic -> scan_service -> exploit

    - v2.5 allows:
        scan_basic -> exploit_bindshell

      if basic scan has already found port 1524.

    This is still realistic because bindshell only requires sufficient port
    evidence, not full service fingerprinting.
    """
    if not state.get("basic_scanned", False):
        return ["scan_basic"]

    actions = []

    # Fast path: if 1524 is visible after basic scan, allow bindshell directly.
    if state.get("has_bindshell_1524", False) and not state.get("has_shell", False):
        actions.append("exploit_bindshell")

    # If service scan has not been done, keep it available for other exploits.
    if not state.get("service_scanned", False):
        actions.append("scan_service")
        return actions

    # After service scan, allow the full exploit set.
    if not state.get("has_shell", False):
        actions.extend(
            [
                "exploit_vsftpd",
                "exploit_unrealircd",
                "exploit_distccd",
                "exploit_samba",
            ]
        )

        if "exploit_bindshell" not in actions:
            actions.append("exploit_bindshell")

    actions.append("stop")

    # De-duplicate while preserving order.
    deduped = []
    for action in actions:
        if action not in deduped:
            deduped.append(action)

    return deduped


# Monkey-patch the environment factory and action mask used by the original
# optimized script.
base_train.make_env = make_env
base_train.get_valid_action_names = get_valid_action_names


if __name__ == "__main__":
    base_train.main()
