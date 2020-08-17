import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LeakyReLU, Dense, Conv2D


# todo: only grayscale so far
# todo: add multiscale conv

class MLP_conv_dense(Model):
    def __init__(self, n_layers_dense_lower=4, n_layers_dense_upper=2, n_layers_conv=None,
                 n_hidden_conv=None, n_hidden_dense_lower=500, n_hidden_dense_lower_output=2, n_hidden_dense_upper=20,
                 spatial_width=28, n_scales=None, n_temporal_basis=10):
        super(MLP_conv_dense, self).__init__()
        self.mlp = MLP(spatial_width, n_hidden_dense_lower_output, n_layers_dense_lower, n_hidden_dense_lower)
        self.conv1 = MultiConv(n_hidden_conv,n_layers_conv)
        self.conv2 = ConvSeq1x1(n_hidden_dense_upper, n_layers_dense_upper, n_temporal_basis)

    def call(self, x):
        x_mlp = self.mlp(x)
        x_conv = self.conv1(x)
        x = tf.concat([x_mlp,x_conv],-1)
        x = self.conv2(x)
        return x


class MLP(Model):

    def __init__(self, spatial_width, n_hidden_dense_lower_output, n_layers, n_hidden):
        super(MLP, self).__init__()
        self.spatial_width = spatial_width
        self.n_hidden_dense_lower_output = n_hidden_dense_lower_output
        self.mlp = [Dense(n_hidden, activation=LeakyReLU()) for i in range(n_layers - 1)]
        self.mlp.append(Dense(n_hidden_dense_lower_output * spatial_width ** 2, activation=LeakyReLU()))

    def call(self, x):
        x = tf.reshape(x, (tf.shape(x)[0], -1))
        for dense in self.mlp:
            x = dense(x)
        x = tf.reshape(x, (-1, self.spatial_width, self.spatial_width, self.n_hidden_dense_lower_output))
        return x


class ConvSeq1x1(Model):
    def __init__(self, channels, n_layers, n_temporal_basis):
        super(ConvSeq1x1, self).__init__()
        self.conv = [Conv2D(channels, 1, activation=LeakyReLU()) for i in range(n_layers - 1)]
        self.conv.append(Conv2D(n_temporal_basis * 2, 1, activation=LeakyReLU()))

    def call(self, x):
        for c in self.conv:
            x = c(x)
        return x


class MultiConv(Model):
    def __init__(self, channels, n_layers):
        super(MultiConv, self).__init__()
        self.conv = [Conv2D(channels, 3, padding='same', activation=LeakyReLU()) for i in range(n_layers)]

    def call(self, x):
        for c in self.conv:
            x = c(x)
        return x
