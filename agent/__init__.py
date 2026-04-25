import gym
import gymnasium
import nasim
from agent.sac import SAC
from agent.softq import SoftQ


def make_agent(env, args):
    obs_dim = env.observation_space.shape[0]

    if isinstance(
        env.action_space,
        (
            gym.spaces.discrete.Discrete,
            gymnasium.spaces.Discrete,
            nasim.envs.action.FlatActionSpace,
        ),
    ):
        print('--> Using Soft-Q agent')
        action_dim = int(env.action_space.n)
        print(type(action_dim))

        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = SoftQ(obs_dim, action_dim, args.train.batch, args)
    else:
        print('--> Using SAC agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]

        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = SAC(obs_dim, action_dim, action_range, args.train.batch, args)

    return agent
