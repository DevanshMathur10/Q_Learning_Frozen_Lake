import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env=gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False,render_mode="rgb_array")

epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#q_table = np.random.uniform(low=-2, high=0, size=[env.observation_space.n,env.action_space.n])
q_table=np.zeros((env.observation_space.n,env.action_space.n))

def show(q_table):
    for i in range(1,len(q_table)+1):
        print(q_table[i-1],end=" ")
        if i%env.action_space.n==0:
            print()

lr=0.01
discount=0.8
episodes=25001
max_steps=10

ep_rewards = []
aggr_ep_rewards={'ep':[],'avg':[],'min':[],'max':[]}

show_every=100

for episode in range(episodes):
    episode_reward=0
    current_state,info=env.reset()
    
    if episode%show_every==0:
        render=True
    else:
        render=False

    for step in range(max_steps):
        if np.random.random()>epsilon:
            #print("Using epsilon.")
            action=np.argmax(q_table[current_state])
        else:
            action=env.action_space.sample()

        #print(current_state,action)

        l =env.step(action)
        new_state=l[0]
        reward=l[1]
        done=l[2]
        episode_reward+=reward

        if render:
            env.render()

        max_feature_q=np.max(q_table[new_state])
        current_q=q_table[current_state,action]

        new_q=(1-lr)*current_q+lr*(reward+discount*max_feature_q)
        q_table[current_state,action]=new_q
    
        if reward==1:
            print(f"We made it on episode {episode} !")

        current_state=new_state
        if done:
            break

    #env.close()
    
    ep_rewards.append(episode_reward)
    #show(q_table)

    if not episode%show_every:
        #np.save(f"MACHINELEARNING/Q-LEARNING/q-tables/{episode}-q-table.npy",q_table)
        average_reward=sum(ep_rewards[-show_every:])/len(ep_rewards[-show_every:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-show_every:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-show_every:]))
        
        print(f"Episode : {episode} Avg : {average_reward} Min: {min(ep_rewards[-show_every:])} Max : {max(ep_rewards[-show_every:])}")


    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['avg'],label='avg')
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['min'],label='min')
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['max'],label='max')

plt.legend(loc=4)
plt.show()
show(q_table)