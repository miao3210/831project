import os 
from stable_baselines3 import PPO, DQN, PPOmiao2, MQN2
import gym

path = './logs-MQN4-min/'
#'./logs-MQN3-min/'
#'./logs-MQN2-min/'
#'./logs-MQN-max/' 
path = path + '/best_model.zip'
#'./logs-PPO2-xi4.0/best_model.zip'
#'./logs-PPO2-seed234/best_model.zip'
#'../midterm-onpolicy/PPO/logs-PPO-again/best_model.zip'
#'../midterm-onpolicy/PPO/logs-basePPO/best_model.zip'
#'./logs/best_model.zip'
#'./logs-DQN-newreward/best_model.zip' 
#'./logs-PPO-newreward/best_model.zip' 
#'./logs-cpu/best_model_cpu'

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
#model = PPOmiao2.load(path)
model = MQN2.load(path)


os.environ['DISPLAY'] = ':1'
rgb_list = []
obs = env.reset()
num_tr = 0
for num_tr in range(8):
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        rgb = env.render()
        #pdb.set_trace()
        rgb_list = rgb_list + rgb 
        if dones:
            env.reset()
            break


import imageio


writer = imageio.get_writer('Cyclic_MQN_psi2_lunarlander-v2.mp4', fps=20)
for k in range(len(rgb_list)):
    writer.append_data(rgb_list[k])
writer.close()