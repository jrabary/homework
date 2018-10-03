import numpy as np


# ========================================================
# 
# Environment-specific cost functions:
#

def cheetah_cost_fn(state, action, next_state):

    front_leg_idx = 14  # 5 gym
    front_shin_idx = 16  # 6 gym
    front_foot_idx = 18  # 7 gym
    x_velocity_idx = 3  # 17 gym
    if len(state.shape) > 1:
        heading_penalty_factor = 10
        scores = np.zeros((state.shape[0],))
        # dont move front shin back so far that you tilt forward
        front_leg = state[:, front_leg_idx]
        my_range = 0.2
        scores[front_leg >= my_range] += heading_penalty_factor

        front_shin = state[:, front_shin_idx]
        my_range = 0
        scores[front_shin >= my_range] += heading_penalty_factor

        front_foot = state[:, front_foot_idx]
        my_range = 0
        scores[front_foot >= my_range] += heading_penalty_factor

        # scores -= (next_state[:, 17] - state[:, 17]) / 0.01  # + 0.1 * (np.sum(action**2, axis=1))
        scores -= state[:, x_velocity_idx]
        return scores

    heading_penalty_factor = 10
    score = 0

    # dont move front shin back so far that you tilt forward
    front_leg = state[front_leg_idx]
    my_range = 0.2
    if front_leg >= my_range:
        score += heading_penalty_factor

    front_shin = state[front_shin_idx]
    my_range = 0
    if front_shin >= my_range:
        score += heading_penalty_factor

    front_foot = state[front_foot_idx]
    my_range = 0
    if front_foot >= my_range:
        score += heading_penalty_factor

    # score -= (next_state[17] - state[17]) / 0.01  # + 0.1 * (np.sum(action**2))
    score -= state[x_velocity_idx]
    return score


# ========================================================
# 
# Cost function for a whole trajectory:
#

def trajectory_cost_fn(cost_fn, states, actions, next_states):
    trajectory_cost = 0
    for i in range(len(actions)):
        trajectory_cost += cost_fn(states[i], actions[i], next_states[i])
    return trajectory_cost
