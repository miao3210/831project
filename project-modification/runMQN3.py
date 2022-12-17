#https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
#https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
#https://stable-baselines.readthedocs.io/en/master/guide/examples.html
import os
import pdb 

import numpy as np 
import matplotlib.pyplot as plt 
import gym # gym version==0.25.0

from stable_baselines3 import MQN3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
#/home/miao/anaconda3/envs/rob831/lib/python3.8/site-packages/stable_baselines3/common/monitor.py 
#/home/miao/anaconda3/envs/rob831/lib/python3.8/site-packages/stable_baselines3/common/results_plotter.py
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        #print(self.n_calls)
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")

          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                if self.num_timesteps % 1000 == 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                #print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

# Create log dir
log_dir = "logs-MQN3-min/"
timesteps = 1e6 * 16 
os.makedirs(log_dir, exist_ok=True)

# Parallel environments
#env = make_vec_env("CartPole-v1", n_envs=4)
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
env = Monitor(env, log_dir)

model = MQN3("MlpPolicy", env, verbose=1, device='cuda', seed=666)
callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=log_dir)
model.learn(total_timesteps=timesteps, callback=callback) #25000)
model.save("mqn3_lunarlander-v2") #cartpole")
plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "MQN3 LunarLander")
#plt.show()
plt.savefig('mqn3_curve.png')
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print('mean reward = {}, std_reward = {}'.format(mean_reward, std_reward))
del model # remove to demonstrate saving and loading

model = PPO.load("mqn3_lunarlander-v2")

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


writer = imageio.get_writer(log_dir + 'mqn3_lunarlander-v2.mp4', fps=20)
for k in range(len(rgb_list)):
    writer.append_data(rgb_list[k])
writer.close()