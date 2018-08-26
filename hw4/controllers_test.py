import gym
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.eager import test
from hw4.controllers import MPCcontroller

tf.enable_eager_execution()


class MPCControllerTest(test_util.TensorFlowTestCase):

    def test_get_action(self):

        env = gym.make('HalfCheetah-v2')
        dyn = tf.keras.layers.Dense(env.observation_space.shape[0])

        controller = MPCcontroller(env, dyn)

        s_0 = env.reset()

        a_0 = controller.get_action(s_0)

        print(a_0)


if __name__ == '__main__':
    test.main()
