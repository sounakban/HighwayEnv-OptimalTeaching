# This code trains a stable_baseline3 DQN


# # Test if GPU is available
# import torch
# if torch.cuda.is_available():
#   print("Found GPU")
#   # Set the default device to GPU 0 if available
#   # torch.set_default_device('cuda:0')
# else:
#   print("No GPU found")




import gymnasium as gym
# from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import RecordVideo
import highway_env
from stable_baselines3 import DQN


# Possible environments and render-modes
# env = gym.make("highway-fast-v0", render_mode="human")  # Nice visualization of learning process
# env = gym.make("highway-fast-v0")
env = gym.make("highway-icy-fast-v0", render_mode="human")
# env = gym.make("highway-icy-fast-v0")

# Record simulation video
## recording needs rgb_array render mode
# env = gym.make("highway-icy-fast-v0", render_mode="rgb_array")
# env = gym.make("highway-fast-v0", render_mode="rgb_array")
## Wrap the env by a RecordVideo wrapper
# env = RecordVideo(env, video_folder="run",
              # episode_trigger=lambda e: True)  # record all episodes
## Provide the video recorder to the wrapped environment
## so it can send it intermediate simulation frames.
# env.unwrapped.set_record_video_wrapper(env)


model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              device="cuda",
              tensorboard_log="highway_dqn/",
              verbose=1)
model.learn(int(2e4))
# model.save("highway_dqn/model")

# Load and test saved model
# model = DQN.load("highway_dqn/model")
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render(render_mode="rgb_array")