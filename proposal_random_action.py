import gym
#from gym.utils.save_video import save_video

import numpy as np 
import matplotlib.pyplot as plt 
import imageio

s = 11

env = gym.make(
    "LunarLander-v2", 
    continuous = False,
    gravity = -10.0,
    enable_wind = False,
    wind_power = 15.0,
    turbulence_power = 1.5,
    new_step_api = False,
    render_mode = 'rgb_array'
)
#https://www.gymlibrary.dev/environments/box2d/lunar_lander/

step_starting_index = 0
episode_index = 0
env.reset()
done = False 
rewards = []
rgb_list = []
actions = []
random_action = 0
t = 0
while not done:
    random_action = env.action_space.sample()
    if t<20:
        #random_action = 0
        t += 1
    observation, reward, done, _ = env.step(random_action)
    #print(reward)
    #print(done)
    rgb = env.render()
    rgb_list = rgb_list + rgb
    rewards.append(reward)
    actions.append(random_action)
    #import pdb
    #pdb.set_trace()
writer = imageio.get_writer('random_agent_'+str(s)+'.mp4', fps=20)
print(len(rgb_list))
for k in range(len(rgb_list)):
    writer.append_data(rgb_list[k])
writer.close()
print(actions)
print(rewards)

#import pdb 
#pdb.set_trace()
'''
save_video(
    env.render(),
    "./",
    fps=env.metadata["render_fps"],
    step_starting_index=step_starting_index,
    episode_index=episode_index
)
'''
rewards = np.array(rewards)
print(rewards.shape)
avg = np.zeros(rewards.shape)
total = np.zeros(rewards.shape)
for k in range(1, rewards.shape[0]+1):
    total[k-1] = rewards[:k].sum()
    avg[k-1] = total[k-1] / k 
#print(total)
#print(avg)

plt.plot(range(rewards.shape[0]), total)
plt.title('LunarLander-v2: total rewards of random agent')
plt.xlabel('steps')
plt.ylabel('total rewards')
plt.savefig('total_'+str(s)+'.pdf')
plt.close()

plt.plot(range(rewards.shape[0]), avg)
plt.title('LunarLander-v2: average rewards of random agent')
plt.xlabel('steps')
plt.ylabel('average rewards')
plt.savefig('avg_'+str(s)+'.pdf')
plt.close()


%fps = 1/self.env.model.opt.timestep