python train_iq.py agent=softq method=iq env=smallhoneypot_simpleQ agent.critic_lr=0.0001 agent.init_temp=1 gamma=0.9 expert.demos=1 env.learn_steps=20000 env.eval_interval=200
#python train_iq.py agent=softq method=iq env=smallhoneypot_simpleQ agent.critic_lr=0.0001 agent.init_temp=1 gamma=0.9 expert.demos=5
python train_iq.py agent=softq method=iq env=smallhoneypot_simpleQ agent.critic_lr=0.0001 agent.init_temp=1 gamma=0.9 expert.demos=10
#python train_iq.py agent=softq method=iq env=smallhoneypot_simpleQ agent.critic_lr=0.0001 agent.init_temp=1 gamma=0.9 expert.demos=20
python train_iq.py agent=softq method=iq env=smallhoneypot_simpleQ agent.critic_lr=0.0001 agent.init_temp=1 gamma=0.9 expert.demos=30
python train_iq.py agent=softq method=iq env=smallhoneypot_simpleQ agent.critic_lr=0.0001 agent.init_temp=1 gamma=0.9 expert.demos=40
python train_iq.py agent=softq method=iq env=smallhoneypot_simpleQ agent.critic_lr=0.0001 agent.init_temp=1 gamma=0.9 expert.demos=50
python train_iq.py agent=softq method=iq env=smallhoneypot_simpleQ agent.critic_lr=0.0001 agent.init_temp=1 gamma=0.9 expert.demos=100
python train_iq.py agent=softq method=iq env=smallhoneypot_simpleQ agent.critic_lr=0.0001 agent.init_temp=1 gamma=0.9 expert.demos=1000
