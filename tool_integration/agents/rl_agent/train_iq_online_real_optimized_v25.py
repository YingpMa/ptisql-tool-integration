from tool_integration.agents.rl_agent import train_iq_online_real_optimized_v2 as base


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
    - v2 usually forces scan_basic -> scan_service -> exploit.
    - v2.5 allows a fast path:
        scan_basic -> exploit_bindshell
      if basic scan already found port 1524.

    This keeps the tool workflow realistic because bindshell only needs
    sufficient port evidence, not full service fingerprinting.
    """
    if not state.get("basic_scanned", False):
        return ["scan_basic"]

    actions = []

    # Fast path: if basic scan found 1524, exploit_bindshell is valid without
    # requiring service_scan.
    if state.get("has_bindshell_1524", False) and not state.get("has_shell", False):
        actions.append("exploit_bindshell")

    # scan_service remains available to support other Metasploit modules.
    if not state.get("service_scanned", False):
        actions.append("scan_service")
        return actions

    if not state.get("has_shell", False):
        actions.extend(
            [
                "exploit_vsftpd",
                "exploit_unrealircd",
                "exploit_distccd",
                "exploit_samba",
            ]
        )

        # Keep bindshell available after service scan too.
        if "exploit_bindshell" not in actions:
            actions.append("exploit_bindshell")

    actions.append("stop")

    # Preserve order and remove duplicates.
    deduped = []
    for a in actions:
        if a not in deduped:
            deduped.append(a)

    return deduped


# Monkey-patch the v2 training module so all existing training/eval logic is reused.
base.make_env = make_env
base.get_valid_action_names = get_valid_action_names


if __name__ == "__main__":
    base.main()
