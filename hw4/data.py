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


def make_sequence_example(states_tm1, actions_tm1, states_t, rewards_t):
    """Convert a trajectory into a SequenceExample.
    """

    # build context feature
    state_dim = tf.train.Feature(int64_list=tf.train.Int64List(value=[states_tm1.shape[1]]))
    action_dim = tf.train.Feature(int64_list=tf.train.Int64List(value=[actions_tm1.shape[1]]))
    length = tf.train.Feature(int64_list=tf.train.Int64List(value=[len(states_tm1)]))
    context = tf.train.Features(feature={'length': length, 'state_dim': state_dim, 'action_dim': action_dim})

    # build feature list
    states_list = [tf.train.Feature(float_list=tf.train.FloatList(value=v)) for v in states_tm1]
    action_list = [tf.train.Feature(float_list=tf.train.FloatList(value=v)) for v in actions_tm1]
    states_tp1_list = [tf.train.Feature(float_list=tf.train.FloatList(value=v)) for v in states_t]
    rewards_tp1_list = [tf.train.Feature(float_list=tf.train.FloatList(value=[v])) for v in rewards_t]

    trajectory = tf.train.FeatureLists(feature_list={
        'states_tm1': tf.train.FeatureList(feature=states_list),
        'actions_tm1': tf.train.FeatureList(feature=action_list),
        'states_t': tf.train.FeatureList(feature=states_tp1_list),
        'rewards_t': tf.train.FeatureList(feature=rewards_tp1_list)
    })

    return tf.train.SequenceExample(context=context, feature_lists=trajectory)


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


def rollout(agent, env, num_trajectories, horizon, output_dir, prefix='train-part'):
    """Sample trajectories using the agent policy"""

    if not (os.path.exists(output_dir)):
        os.makedirs(output_dir)

    num_splits = len([name for name in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, name))])

    writer = tf.python_io.TFRecordWriter(os.path.join(output_dir, f"{prefix}-{num_splits}.tfr"))

    for j in range(num_trajectories):
        obs_tm1 = env.reset()

        observations = [np.reshape(obs_tm1, [1, -1])]
        actions = []
        rewards = []
        for t in range(horizon):
            a_tm1 = agent.get_action(obs_tm1)
            actions.append(np.reshape(a_tm1, [1, -1]))

            obs_t, r_t, done, info = env.step(a_tm1)
            observations.append(np.reshape(obs_t, [1, -1]))
            rewards.append(r_t)
            obs_tm1 = obs_t

        observations = np.concatenate(observations).astype(np.float32)
        actions = np.concatenate(actions)
        rewards = np.array(rewards)
        ex = make_sequence_example(observations[:-1], actions, observations[1:], rewards)
        writer.write(ex.SerializeToString())

    writer.close()


def make_trajectory_dataset(data_files, state_dim, action_dim):
    """
    Create a dataset of trajectories from a list of TFRecords containing trajectory sequence example.
    Args:
        data_files: lis of TFRecords containing SequenceExample of trajectory
        state_dim: dimension of the state
        action_dim: dimension of the action

    Returns:
        tf.Dataset of trajectories
    """
    data = (tf.data.Dataset.list_files(data_files)
            .interleave(tf.data.TFRecordDataset, 1)
            .map(TrajectoryDecoder(state_dim, action_dim)))

    return data


def make_transition_dataset(data_files, state_dim, action_dim):
    """
    Create a data set of transition (x_tm1, a_tm1, x_t, r_t) from a list of trajectory SequenceExample
    Args:
        data_files: list of TFRecords files containing SequenceExample of trajectory
        state_dim: dimension of the state
        action_dim: dimension of the action

    Returns:
        tf.Dataset of transition
    """
    data = (tf.data.Dataset.list_files(data_files)
            .interleave(tf.data.TFRecordDataset, 1)
            .map(TrajectoryDecoder(state_dim, action_dim))
            .flat_map(tf.data.Dataset.from_tensor_slices))

    return data


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
    return state, action, next_state # tf.subtract(next_state, state)


class TrajectoryDecoder(object):
    """
    Decode trajectory stored as a SequenceExample
    """

    def __init__(self, state_dim, action_dim):

        # Define how to parse the example
        self._context_features = {
            "length": tf.FixedLenFeature([], dtype=tf.int64),
            "state_dim": tf.FixedLenFeature([], dtype=tf.int64),
            "action_dim": tf.FixedLenFeature([], dtype=tf.int64),
        }

        self._sequence_features = {
            "states_tm1": tf.FixedLenSequenceFeature([state_dim], dtype=tf.float32),
            "actions_tm1": tf.FixedLenSequenceFeature([action_dim], dtype=tf.float32),
            "states_t": tf.FixedLenSequenceFeature([state_dim], dtype=tf.float32),
            "rewards_t": tf.FixedLenSequenceFeature([], dtype=tf.float32),
        }

    def __call__(self, example_serialized):

        # Parse the examplese
        _, sequence_parsed = tf.parse_single_sequence_example(
            serialized=example_serialized,
            context_features=self._context_features,
            sequence_features=self._sequence_features
        )

        return sequence_parsed
