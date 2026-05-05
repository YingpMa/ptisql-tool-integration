"""
Optimized v2.5-stable training/evaluation entry point.

Place this file at:
  tool_integration/agents/rl_agent/train_iq_online_real_optimized_v25_stable.py

It reuses the original optimized training script and switches:
1. environment -> pt_env_optimized_v25_stable.RealPTEnv
2. action mask -> stable fast-path bindshell

Difference from v2.5:
- v2.5 allows scan_basic -> exploit_bindshell when port 1524 is visible.
- v2.5-stable allows this fast path only before any failed attempt.
- If the fast-path attempt fails, the agent is forced back to scan_service.

This keeps the speed advantage of early bindshell attempts while reducing
repeated failed fast-path attempts.
"""

import tool_integration.agents.rl_agent.train_iq_online_real_optimized as base_train


def make_env(args):
    from tool_integration.agents.rl_agent.pt_env_optimized_v25_stable import RealPTEnv

    return RealPTEnv(
        target_ip=args.target_ip,
        use_metasploit=args.use_metasploit,
    )


def get_valid_action_names(state):
    """
    v2.5-stable action mask.

    Fast-path bindshell is allowed only as the first exploit attempt.
    If that attempt fails, failed_attempts becomes > 0 and the agent is
    forced back to scan_service before trying the full exploit set.
    """
    if not state.get("basic_scanned", False):
        return ["scan_basic"]

    failed_attempts = int(state.get("failed_attempts", 0))
    has_shell = bool(state.get("has_shell", False))
    service_scanned = bool(state.get("service_scanned", False))
    has_bindshell = bool(state.get("has_bindshell_1524", False))

    actions = []

    # Fast path: only allow before service scan and only if no exploit has failed.
    if has_bindshell and not has_shell and not service_scanned and failed_attempts == 0:
        actions.append("exploit_bindshell")

    # If service scan has not been done, keep it available.
    # After a failed fast-path attempt, scan_service becomes the only valid
    # next action.
    if not service_scanned:
        actions.append("scan_service")
        return actions

    # After service scan, allow the full exploit set.
    if not has_shell:
        actions.extend(
            [
                "exploit_vsftpd",
                "exploit_unrealircd",
                "exploit_distccd",
                "exploit_samba",
            ]
        )

        if has_bindshell:
            actions.append("exploit_bindshell")

    actions.append("stop")

    # De-duplicate while preserving order.
    deduped = []
    for action in actions:
        if action not in deduped:
            deduped.append(action)

    return deduped


# Monkey-patch the original optimized training script.
base_train.make_env = make_env
base_train.get_valid_action_names = get_valid_action_names


if __name__ == "__main__":
    base_train.main()
