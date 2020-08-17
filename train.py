import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from model import DiffusionModel
import numpy as np

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

datasets, datasets_info = tfds.load(name='mnist',
                                    with_info=True,
                                    as_supervised=False)

n_colors = 1
spatial_width = 28
batch_size = 512
lr=1e-3

model = DiffusionModel()

def _preprocess(sample):
    image = tf.cast(sample['image'], tf.float32) / 255. -0.5# Scale to unit interval.
    return image




train_dataset = (datasets['train']
                 .map(_preprocess)
                 .batch(batch_size)
                 .prefetch(tf.data.experimental.AUTOTUNE)
                 .shuffle(int(10e3)))

eval_dataset = (datasets['test']
                .map(_preprocess)
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE))


model.train(train_dataset,epochs=1000)
model.save()
model.load()
#model.generate_samples()


