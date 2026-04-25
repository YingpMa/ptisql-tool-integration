from real_env_demo import RealPentestEnv

env = RealPentestEnv()

obs, info = env.reset()
print("reset:", obs, info)

obs, reward, terminated, truncated, info = env.step(0)
print("after scan:", obs, reward, terminated, truncated)

obs, reward, terminated, truncated, info = env.step(1)
print("after exploit:", obs, reward, terminated, truncated)
