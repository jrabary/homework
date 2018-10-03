import gym
import pickle
import numpy as np
import pybullet
import pybullet_envs
import tensorflow as tf
from hw4.dynamics import build_mlp_2
from hw4.controllers import MPCcontroller
# from hw4.cheetah_env import HalfCheetahEnvNew
from hw4.cost_functions import cheetah_cost_fn

tf.enable_eager_execution()

tfe = tf.contrib.eager

# pybullet.connect(pybullet.DIRECT)

env = gym.make("HalfCheetahBulletEnv-v0")
env.render(mode='human')
# env = HalfCheetahEnvNew()

with open('hw4/data/half_cheetah_bullet/moments.pkl', 'rb') as f:
    moments = pickle.load(f)

print(moments)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
print('observation dim', obs_dim)
dyn_model = build_mlp_2([500, 500], obs_dim)

# init model
dyn_model(tf.random_uniform([1, obs_dim + act_dim]))
checkpoint = tfe.Checkpoint(dynamics=dyn_model)

checkpoint_dir = 'hw4/model_dir'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
print(f"Load checkpoint {checkpoint_path}")
checkpoint.restore(checkpoint_path)

controller = MPCcontroller(env, dyn_model, moments=moments, cost_fn=cheetah_cost_fn, horizon=15, num_simulated_paths=1000)

observation = env.reset()

for _ in range(1000):
    action = controller.get_action(observation)
    # print(act)
    # action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    env.render(mode='human')
