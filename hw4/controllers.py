import tensorflow as tf
import numpy as np
import numpy.random as npr
from hw4.cost_functions import trajectory_cost_fn
import time


class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        """ YOUR CODE HERE """
        self.action_space = env.action_space
        pass

    def get_action(self, state):
        """ Your code should randomly sample an action uniformly from the action space """
        return self.action_space.sample()


class MPCcontroller(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """

    def __init__(self,
                 env,
                 dyn_model,
                 horizon=5,
                 cost_fn=None,
                 num_simulated_paths=10,
                 ):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Note: be careful to batch your simulations through the model for speed """

        # action_samples = tf.random_uniform((self.num_simulated_paths, self.horizon, self.env.action_space.shape[0]),
        #                                    minval=self.env.action_space.low,
        #                                    maxval=self.env.action_space.high)

        action_samples = npr.uniform(self.env.action_space.low, self.env.action_space.high,
                                  (self.num_simulated_paths, self.horizon, self.env.action_space.shape[0]))

        # state = tf.convert_to_tensor(state.astype(np.float32))
        state = np.reshape(state, [1, -1])
        state = np.tile(state, [self.num_simulated_paths, 1])  # KxHxD_S

        states = [np.expand_dims(state, axis=1)]

        # t0 = time.time()
        for t in range(self.horizon):
            cur_inputs = np.concatenate([state, action_samples[:, t, :]], axis=1).astype(np.float32) # KxHxD
            states_diff = self.dyn_model(tf.convert_to_tensor(cur_inputs)).numpy()  # KxD_S
            state += states_diff  # KxD_S
            states.append(np.expand_dims(state, axis=1))

        states = tf.concat(states, axis=1).numpy()
        # print('forward', time.time() - t0)
        # action_samples = action_samples.numpy()
        costs = np.zeros((self.num_simulated_paths,))

        t0 = time.time()

        for t in range(self.horizon-1):
            costs += self.cost_fn(states[:, t, :], action_samples[:, t, 0], states[:, t+1, :])

        min_j = np.argmin(costs)
        # print('min', time.time() - t0)

        return action_samples[min_j, 0, :]

