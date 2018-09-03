import gym
import numpy as np
import tensorflow as tf
from hw4.dynamics import build_mlp_2
from hw4.controllers import MPCcontroller
from hw4.cheetah_env import HalfCheetahEnvNew
from hw4.cost_functions import cheetah_cost_fn

tf.enable_eager_execution()


tfe = tf.contrib.eager

env = HalfCheetahEnvNew()

obs_dim = env.observation_space.shape[0]
dyn_model = build_mlp_2([100, 100], obs_dim)

controller = MPCcontroller(env, dyn_model, cost_fn=cheetah_cost_fn, horizon=15, num_simulated_paths=1000)

checkpoint = tfe.Checkpoint(dynamics=dyn_model)

checkpoint_dir = 'model_dir'

observation = env.reset()

# init model
dyn_model(tf.random_uniform([1, 26]))
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


for _ in range(1000):
  env.render()

  action = controller.get_action(observation)
  # print(act)
  # action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)