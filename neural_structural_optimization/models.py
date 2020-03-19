# lint as python3
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=missing-docstring
# pylint: disable=invalid-name

import autograd
import autograd.core
import autograd.numpy as np
from neural_structural_optimization import topo_api
from neural_structural_optimization import problems
import tensorflow as tf

# requires tensorflow 2.0

layers = tf.keras.layers


def batched_topo_loss(params, envs):
    losses = [env.objective(params[i], volume_contraint=True)
              for i, env in enumerate(envs)]
    return np.stack(losses)


def convert_autograd_to_tensorflow(func):
    @tf.custom_gradient
    def wrapper(x):
        vjp, ans = autograd.core.make_vjp(func, x.numpy())
        return ans, vjp
    
    return wrapper


def set_random_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)


class Model(tf.keras.Model):
    
    def __init__(self, seed=None, args=None):
        super().__init__()
        set_random_seed(seed)
        self.seed = seed
        self.env = topo_api.Environment(args)
    
    def loss(self, logits):
        # for our neural network, we use float32, but we use float64 for the physics
        # to avoid any chance of overflow.
        # add 0.0 to work-around bug in grad of tf.cast on NumPy arrays
        logits = 0.0 + tf.cast(logits, tf.float64)
        f = lambda x: batched_topo_loss(x, [self.env])
        losses = convert_autograd_to_tensorflow(f)(logits)
        return tf.reduce_mean(losses)


class PixelModel(Model):
    
    def __init__(self, seed=None, args=None):
        super().__init__(seed, args)
        shape = (1, self.env.args['nely'], self.env.args['nelx'])
        z_init = np.broadcast_to(args['volfrac'] * args['mask'], shape)
        self.z = tf.Variable(z_init, trainable=True)
    
    def call(self, inputs=None):
        return self.z


def global_normalization(inputs, epsilon=1e-6):
    mean, variance = tf.nn.moments(inputs, axes=list(range(len(inputs.shape))))
    net = inputs
    net -= mean
    net *= tf.math.rsqrt(variance + epsilon)
    return net


def UpSampling2D(factor):
    return layers.UpSampling2D((factor, factor), interpolation='bilinear')


def Conv2D(filters, kernel_size, **kwargs):
    return layers.Conv2D(filters, kernel_size, padding='same', **kwargs)


class AddOffset(layers.Layer):
    
    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale
    
    def build(self, input_shape):
        self.bias = self.add_weight(
            shape=input_shape, initializer='zeros', trainable=True, name='bias')
    
    def call(self, inputs):
        return inputs + self.scale * self.bias


class CNNModel(Model):
    
    def __init__(
            self,
            seed=0,
            args=None,
            latent_size=128,
            dense_channels=32,
            resizes=(1, 2, 2, 2, 1),
            conv_filters=(128, 64, 32, 16, 1),
            offset_scale=10,
            kernel_size=(5, 5),
            latent_scale=1.0,
            dense_init_scale=1.0,
            activation=tf.nn.tanh,
            conv_initializer=tf.initializers.VarianceScaling,
            normalization=global_normalization,
    ):
        super().__init__(seed, args)
        
        if len(resizes) != len(conv_filters):
            raise ValueError('resizes and filters must be same size')
        
        activation = layers.Activation(activation)
        
        total_resize = int(np.prod(resizes))
        h = self.env.args['nely'] // total_resize
        w = self.env.args['nelx'] // total_resize
        
        net = inputs = layers.Input((latent_size,), batch_size=1)
        filters = h * w * dense_channels
        dense_initializer = tf.initializers.orthogonal(
            dense_init_scale * np.sqrt(max(filters / latent_size, 1)))
        net = layers.Dense(filters, kernel_initializer=dense_initializer)(net)
        net = layers.Reshape([h, w, dense_channels])(net)
        
        for resize, filters in zip(resizes, conv_filters):
            net = activation(net)
            net = UpSampling2D(resize)(net)
            net = normalization(net)
            net = Conv2D(
                filters, kernel_size, kernel_initializer=conv_initializer)(net)
            if offset_scale != 0:
                net = AddOffset(offset_scale)(net)
        
        outputs = tf.squeeze(net, axis=[-1])
        self.core_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        latent_initializer = tf.initializers.RandomNormal(stddev=latent_scale)
        self.z = self.add_weight(
            shape=inputs.shape, initializer=latent_initializer, name='z')
    
    def call(self, inputs=None):
        return self.core_model(self.z)


def MaxPooling():
    return layers.MaxPool2D(pool_size=(2,2), strides=(2, 2))

def deconv(x, filters):
    x = layers.Conv2DTranspose(filters=filters, kernel_size=3, strides=(2,2), padding='same')(x)
    x = tf.keras.activations.relu(x)
    return global_normalization(x)

class FCNModel(Model):
    def __init__(
            self,
            seed=0,
            args=None,
            latent_size=128,
            dense_channels=32,
            resizes=(1, 2, 2, 2, 1),
            latent_scale=1.0,
            dense_init_scale=1.0):
        super().__init__(seed, args)
        
        total_resize = 1   # 避免max pool出现负数大小
        h = self.env.args['nely'] // total_resize
        w = self.env.args['nelx'] // total_resize

        net = inputs = layers.Input((latent_size,), batch_size=1)
        filters = h * w * dense_channels
        dense_initializer = tf.initializers.orthogonal(
            dense_init_scale * np.sqrt(max(filters / latent_size, 1)))
        net = layers.Dense(filters, kernel_initializer=dense_initializer)(net)
        net = layers.Reshape([h, w, dense_channels])(net)
        
        # ------------ 上同CNN
        # VGG
        vgg_output = {}
        features = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]
        for index, (repeat, filters) in enumerate(features):
            for i in range(repeat):
                net = Conv2D(filters=filters, kernel_size=3)(net)
                net = layers.ReLU()(net)
            net = MaxPooling()(net)
            vgg_output[f"x{index + 1}"] = net
        # FCN
        x5 = vgg_output['x5']
        x4 = vgg_output['x4']
        x3 = vgg_output['x3']
        net = deconv(x5, 512)
        net = deconv(net + x4, 256)
        net = deconv(net + x3, 128)
        net = deconv(net, 64)
        net = deconv(net, 32)
        # Classifier
        net = Conv2D(1, 1)(net)
        net = tf.keras.activations.sigmoid(net)
        
        # ------------ 下同CNN
        outputs = tf.squeeze(net, axis=[-1])
        self.core_model = tf.keras.Model(inputs=inputs, outputs=outputs)

        latent_initializer = tf.initializers.RandomNormal(stddev=latent_scale)
        self.z = self.add_weight(
            shape=inputs.shape, initializer=latent_initializer, name='z')
    
    def call(self, inputs):
        return self.core_model(self.z)


class UNetModel(Model):
    def __init__(
            self,
            seed=0,
            args=None,
            latent_size=128,
            dense_channels=32,
            latent_scale=1.0,
            dense_init_scale=1.0):
        super().__init__(seed, args)
        
        total_resize = 1  # 避免max pool出现负数大小
        h = self.env.args['nely'] // total_resize
        w = self.env.args['nelx'] // total_resize
        
        net = inputs = layers.Input((latent_size,), batch_size=1)
        filters = h * w * dense_channels
        dense_initializer = tf.initializers.orthogonal(
            dense_init_scale * np.sqrt(max(filters / latent_size, 1)))
        net = layers.Dense(filters, kernel_initializer=dense_initializer)(net)
        net = layers.Reshape([h, w, dense_channels])(net)
        
        # ------------ 上同CNN
        # contract
        channels = (64, 128, 256, 512)
        contract_output = {}
        for index, channel in enumerate(channels):
            net = Conv2D(channel, 3)(net)
            net = layers.ReLU()(net)
            net = Conv2D(channel, 3)(net)
            net = layers.ReLU()(net)
            contract_output[F'x{index + 1}'] = net
            net = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(net)

        net = Conv2D(1024, 3)(net)
        net = layers.ReLU()(net)
        net = Conv2D(1024, 3)(net)
        net = layers.ReLU()(net)
        
        channels = (512, 256, 128, 64)
        for index, channel in enumerate(channels):
            net = layers.UpSampling2D(interpolation='bilinear')(net)
            net = Conv2D(channel, 2)(net)

            x = contract_output[F'x{len(channels) - index}']
            net = layers.concatenate([x, net], axis=3)
            net = Conv2D(channel, 3)(net)
            net = layers.ReLU()(net)
            net = Conv2D(channel, 3)(net)
            net = layers.ReLU()(net)
        
        net = Conv2D(1, 1)(net)
        # ------------ 下同CNN
        outputs = tf.squeeze(net, axis=[-1])
        self.core_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        latent_initializer = tf.initializers.RandomNormal(stddev=latent_scale)
        self.z = self.add_weight(
            shape=inputs.shape, initializer=latent_initializer, name='z')
    
    def call(self, inputs):
        return self.core_model(self.z)

if __name__ == '__main__':
    problem = problems.mbb_beam(height=64, width=128)
    args = topo_api.specified_task(problem)
    p = UNetModel(args=args)
    print(p)