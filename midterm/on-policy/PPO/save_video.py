import os 
from stable_baselines3 import PPO
import gym

path = './logs-cpu/best_model_cpu'
model = PPO.load(path)

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


writer = imageio.get_writer('PPO_lunarlander-v2-cpu.mp4', fps=20)
for k in range(len(rgb_list)):
    writer.append_data(rgb_list[k])
writer.close()