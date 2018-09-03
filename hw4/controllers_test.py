import gym
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.eager import test
from hw4.controllers import MPCcontroller
from hw4.cost_functions import cheetah_cost_fn
from hw4.cheetah_env import HalfCheetahEnvNew
tf.enable_eager_execution()


class MPCControllerTest(test_util.TensorFlowTestCase):

    def test_get_action(self):

        # env = gym.make('HalfCheetah-v2')
        env = HalfCheetahEnvNew()
        dyn = tf.keras.layers.Dense(env.observation_space.shape[0])

        controller = MPCcontroller(env, dyn, cost_fn=cheetah_cost_fn, num_simulated_paths=18, horizon=5)

        s_0 = env.reset()

        for _ in range(10):
            a_0 = controller.get_action(s_0)
            print(a_0)



if __name__ == '__main__':
    test.main()
