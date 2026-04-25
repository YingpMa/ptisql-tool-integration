from tool_integration.agents.rl_agent.pt_env import RealPTEnv

env = RealPTEnv(
    target_ip="10.11.202.189",
    use_metasploit=True
)

state = env.reset()
print("reset:", state)

actions = [0, 1, 3]

for action in actions:
    next_state, reward, done, info = env.step(action)

    print("action =", action, env.ACTIONS[action])
    print("reward =", reward)
    print("done =", done)
    print("info =", info)
    print("next_state =", next_state)
    print("-" * 50)

    if done:
        break