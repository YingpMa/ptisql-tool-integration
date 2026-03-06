#python train_rl.py agent=softq method=iq env=smallhoneypot expert.demos=1 expert.subsample_freq=20 agent.init_temp=0.001 method.chi=True method.loss=value_expert num_seed_steps=1000
#python train_rl.py agent=softq method=iq env=smallhoneypot_simpleQ agent.critic_lr=0.0001 agent.init_temp=1 gamma=0.9 env.eval_interval=1e3 env.learn_steps=1e5 
python train_rl.py agent=softq method=iq env=smallhoneypot_simpleQ agent.critic_lr=0.0001 agent.init_temp=1 gamma=0.9 env.eval_interval=1e3 env.learn_steps=1e5
