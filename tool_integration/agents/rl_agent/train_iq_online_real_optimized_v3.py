from tool_integration.agents.rl_agent import train_iq_online_real_optimized_v2 as base


def make_env(args):
    from tool_integration.agents.rl_agent.pt_env_optimized_v3 import RealPTEnv

    return RealPTEnv(
        target_ip=args.target_ip,
        use_metasploit=args.use_metasploit,
    )


base.make_env = make_env


if __name__ == "__main__":
    base.main()
