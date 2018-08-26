import tensorflow as tf
import numpy as np
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

        action_samples = tf.random_uniform((self.num_simulated_paths, self.horizon, self.env.action_space.shape[0]),
                                           minval=self.env.action_space.low,
                                           maxval=self.env.action_space.high)

        state = tf.convert_to_tensor(state.astype(np.float32))
        state = tf.reshape(state, [1, self.env.observation_space.shape[0]])
        state = tf.tile(state, [self.num_simulated_paths, 1])  # Kx1xD_S
        # print(state.shape)

        states = [tf.expand_dims(state, axis=1)]

        for t in range(self.horizon):

            cur_inputs = tf.concat([state, action_samples[:, t, :]], axis=1)  # KxHxD
            states_diff = self.dyn_model(cur_inputs)  # KxD_S
            next_states = state + states_diff  # KxD_S
            states.append(tf.expand_dims(next_states, axis=1))
            # print(next_states.shape)

        states = tf.concat(states, axis=1)



        print(states.shape)
        return self.env.action_space.sample()

