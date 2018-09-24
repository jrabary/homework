import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from hw4.data import decode_example


tf.enable_eager_execution()

batch_size = 100

data = (tf.data.Dataset.list_files('data/half_cheetah-new/part-*')
        .interleave(tf.data.TFRecordDataset, 1)
        .map(decode_example)
        .batch(batch_size)
        )

state_scaler = StandardScaler()
act_scaler = StandardScaler()
delta_scaler = StandardScaler()

for state, act, delta in data:
    state_scaler.partial_fit(state)
    act_scaler.partial_fit(act)
    delta_scaler.partial_fit(delta)

print(act_scaler.mean_)
print(act_scaler.var_)
