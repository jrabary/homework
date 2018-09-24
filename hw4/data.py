import os, os.path
import tensorflow as tf
import numpy as np
import time

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _to_tf_example(state, action, next_state):

    state = state.tolist()
    action = action.tolist()
    next_state = next_state.tolist()

    """Convert state-action transition into tf_example"""
    example = tf.train.Example(features=tf.train.Features(feature={
        'state': _float_feature(state),
        'action': _float_feature(action),
        'next_state': _float_feature(next_state)}))
    return example


def save_examples(output_dir, states, actions, next_states, prefix='part'):
    """Save trajectory rollout into tfrecords"""
    if not (os.path.exists(output_dir)):
        os.makedirs(output_dir)

    num_splits = len([name for name in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, name))])

    writer = tf.python_io.TFRecordWriter(os.path.join(output_dir, f"{prefix}-{num_splits}.tfr"))

    for i in range(states.shape[0]):
        example = _to_tf_example(states[i], actions[i], next_states[i])
        writer.write(example.SerializeToString())
    writer.close()


def rollout(agent, env, num_trajectories, horizon, output_dir):
    """Sample trajectories using the agent policy"""
    observations = []
    actions = []
    next_observations = []
    for j in range(num_trajectories):
        obs_t = env.reset()

        # t0 = time.time()
        for t in range(horizon):
            a_t = agent.get_action(obs_t)
            # print(f'path {j} {t}', flush=True)
            obs_t_plus_1, reward, done, info = env.step(a_t)
            observations.append(np.reshape(obs_t, [1, -1]))
            actions.append(np.reshape(a_t, [1, -1]))
            next_observations.append(np.reshape(obs_t_plus_1, (1, -1)))
        # print('rollout step', time.time() - t0)

    observations = np.concatenate(observations).astype(np.float32)

    actions = np.concatenate(actions)

    next_observations = np.concatenate(next_observations)

    save_examples(output_dir, observations, actions, next_observations)


def decode_example(example_serialized):
    """Decode tfrecords example"""

    feature_map = {
        'state': tf.VarLenFeature(dtype=tf.float32),
        'action': tf.VarLenFeature(dtype=tf.float32),
        'next_state': tf.VarLenFeature(dtype=tf.float32)
    }

    features = tf.parse_single_example(example_serialized, feature_map)

    state = features['state'].values
    action = features['action'].values
    next_state = features['next_state'].values

    #return tf.concat([state, action], axis=0), tf.subtract(next_state, state)
    return state, action, tf.subtract(next_state, state)
