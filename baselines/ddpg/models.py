import tensorflow as tf
import tensorflow.contrib as tc
import os
import tensorflow.contrib.layers as layers


def tiny_yolo(X):
    '''
    0 conv     16  3 x 3 / 1   224 x 224 x   3   ->   224 x 224 x  16
    1 max          2 x 2 / 2   224 x 224 x  16   ->   112 x 112 x  16
    2 conv     32  3 x 3 / 1   112 x 112 x  16   ->   112 x 112 x  32
    3 max          2 x 2 / 2   112 x 112 x  32   ->    56 x  56 x  32
    4 conv     16  1 x 1 / 1    56 x  56 x  32   ->    56 x  56 x  16
    5 conv    128  3 x 3 / 1    56 x  56 x  16   ->    56 x  56 x 128
    6 conv     16  1 x 1 / 1    56 x  56 x 128   ->    56 x  56 x  16
    7 conv    128  3 x 3 / 1    56 x  56 x  16   ->    56 x  56 x 128
    8 max          2 x 2 / 2    56 x  56 x 128   ->    28 x  28 x 128
    9 conv     32  1 x 1 / 1    28 x  28 x 128   ->    28 x  28 x  32
   10 conv    256  3 x 3 / 1    28 x  28 x  32   ->    28 x  28 x 256
   11 conv     32  1 x 1 / 1    28 x  28 x 256   ->    28 x  28 x  32
   12 conv    256  3 x 3 / 1    28 x  28 x  32   ->    28 x  28 x 256
   13 max          2 x 2 / 2    28 x  28 x 256   ->    14 x  14 x 256
   14 conv     64  1 x 1 / 1    14 x  14 x 256   ->    14 x  14 x  64
   15 conv    512  3 x 3 / 1    14 x  14 x  64   ->    14 x  14 x 512
   16 conv     64  1 x 1 / 1    14 x  14 x 512   ->    14 x  14 x  64
   17 conv    512  3 x 3 / 1    14 x  14 x  64   ->    14 x  14 x 512
   18 conv    128  1 x 1 / 1    14 x  14 x 512   ->    14 x  14 x 128
   19 conv   1000  1 x 1 / 1    14 x  14 x 128   ->    14 x  14 x1000
   20 avg                       14 x  14 x1000   ->  1000
   21 softmax                                        1000
   22 cost                                           1000
    '''
    conv0 = tf.layers.conv2d(inputs=X, filters=16, kernel_size=[3, 3], strides=1, padding="same", activation=tf.nn.relu)
    max1 = tf.layers.max_pooling2d(inputs=conv0, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=max1, filters=32, kernel_size=[3, 3], strides=1, padding="same",
                             activation=tf.nn.relu)
    max3 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    conv4 = tf.layers.conv2d(inputs=max3, filters=16, kernel_size=[1, 1], strides=1, padding="same",
                             activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(inputs=conv4, filters=128, kernel_size=[3, 3], strides=1, padding="same",
                             activation=tf.nn.relu)
    conv6 = tf.layers.conv2d(inputs=conv5, filters=16, kernel_size=[1, 1], strides=1, padding="same",
                             activation=tf.nn.relu)
    conv7 = tf.layers.conv2d(inputs=conv6, filters=128, kernel_size=[3, 3], strides=1, padding="same",
                             activation=tf.nn.relu)
    max8 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)
    conv9 = tf.layers.conv2d(inputs=max8, filters=32, kernel_size=[1, 1], strides=1, padding="same",
                             activation=tf.nn.relu)
    conv10 = tf.layers.conv2d(inputs=conv9, filters=256, kernel_size=[3, 3], strides=1, padding="same",
                              activation=tf.nn.relu)
    conv11 = tf.layers.conv2d(inputs=conv10, filters=32, kernel_size=[1, 1], strides=1, padding="same",
                              activation=tf.nn.relu)
    conv12 = tf.layers.conv2d(inputs=conv11, filters=256, kernel_size=[3, 3], strides=1, padding="same",
                              activation=tf.nn.relu)
    max13 = tf.layers.max_pooling2d(inputs=conv12, pool_size=[2, 2], strides=2)
    conv14 = tf.layers.conv2d(inputs=max13, filters=64, kernel_size=[1, 1], strides=1, padding="same",
                              activation=tf.nn.relu)
    conv15 = tf.layers.conv2d(inputs=conv14, filters=512, kernel_size=[3, 3], strides=1, padding="same",
                              activation=tf.nn.relu)
    conv16 = tf.layers.conv2d(inputs=conv15, filters=64, kernel_size=[1, 1], strides=1, padding="same",
                              activation=tf.nn.relu)
    conv17 = tf.layers.conv2d(inputs=conv16, filters=512, kernel_size=[3, 3], strides=1, padding="same",
                              activation=tf.nn.relu)
    conv18 = tf.layers.conv2d(inputs=conv17, filters=128, kernel_size=[1, 1], strides=1, padding="same",
                              activation=tf.nn.relu)
    conv19 = tf.layers.conv2d(inputs=conv18, filters=1000, kernel_size=[1, 1], strides=1, padding="same",
                              activation=tf.nn.relu)
    avg20 = tf.layers.average_pooling2d(inputs=conv19, pool_size=[conv19.shape[1], conv19.shape[2]], strides=1)
    return layers.flatten(avg20)

class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm


    def __call__(self, obs, reuse=False):
        drop_out = True
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = obs
            (v, o, r, rgbd) = tf.split(out, [1, 3, 3, int(out.shape[1])-7], 1)
            #rgbd = tf.reshape(rgbd, [-1, 288, 256, 4])
            rgbd = tf.reshape(rgbd, [-1, 144, 256, 4])

            #(rgbd0, rgbd1) = tf.split(rgbd, [144, 144], 1)
            (rgb0, depth0) = tf.split(rgbd, [3,1], 3)
            #(rgb1, depth1) = tf.split(rgbd1, [3,1], 3)
            '''
            with tf.variable_scope("convnet"):
                for num_outputs, kernel_size, stride in convs:
                rgb0 = layers.convolution2d(rgb0,
                       num_outputs=num_outputs,
                       kernel_size=kernel_size,
                       stride=stride,
                       activation_fn=tf.nn.relu)
                rgb0 = tf.layers.batch_normalization(rgb0)
                for num_outputs, kernel_size, stride in convs:
                rgb1 = layers.convolution2d(rgb1,
                       num_outputs=num_outputs,
                       kernel_size=kernel_size,
                       stride=stride,
                       activation_fn=tf.nn.relu)
                rgb1 = tf.layers.batch_normalization(rgb1)
            '''
            '''
                for num_outputs, kernel_size, stride in convs:
                depth0 = layers.convolution2d(depth0,
                       num_outputs=num_outputs,
                       kernel_size=kernel_size,
                       stride=stride,
                       activation_fn=tf.nn.relu)
                depth0 = tf.layers.batch_normalization(depth0)
                for num_outputs, kernel_size, stride in convs:
                depth1 = layers.convolution2d(depth1,
                       num_outputs=num_outputs,
                       kernel_size=kernel_size,
                       stride=stride,
                       activation_fn=tf.nn.relu)
                depth1 = tf.layers.batch_normalization(depth1)
                '''
            rgb0_out = tiny_yolo(rgb0)
            #rgb1_out = tiny_yolo(rgb1)
            depth0_out = tiny_yolo(depth0)
            #depth1_out = tiny_yolo(depth1)

            #rgb0_out = layers.flatten(rgb0)
            #rgb1_out = layers.flatten(rgb1)
            #depth0_out = layers.flatten(depth0)
            #depth1_out = layers.flatten(depth1)
            x = tf.concat([v, o, r, rgb0_out, depth0_out], 1)
            #conv_out = tf.concat([rgb0_out, rgb1_out], axis=1)
            #conv_out = layers.flatten(out)
            x = tf.layers.dense(x, 2048)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 2048)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


    def call_old(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            (last_act, plane_box, bird_box, maybe_plane_box, maybe_bird_box, rgbd) = tf.split(x, [2, 4, 4, 4, 4, int(x.shape[1])-(2+4+4+4+4)], 1)
            rgbd = tf.reshape(rgbd, [-1, 144, 256, 4])
            # NEW
            # For 2 Frames of RGB
            #(rgb0, rgb1) = tf.split(x, [144, 144], 1)
            (rgb0, depth0) = tf.split(rgbd, [3,1], 3)
            #(rgb1, depth1) = tf.split(rgb1, [3,1], 3)
            rgb0 = tf.layers.conv2d(inputs=rgb0,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
            rgb0 = tf.layers.max_pooling2d(inputs=rgb0, pool_size=[2, 2], strides=2)
            rgb0 = tf.layers.conv2d(inputs=rgb0,filters=32,kernel_size=[4, 4],padding="same",activation=tf.nn.relu)
            rgb0 = tf.layers.max_pooling2d(inputs=rgb0, pool_size=[2, 2], strides=2)
            rgb0 = tf.layers.conv2d(inputs=rgb0,filters=32,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
            rgb0 = tf.layers.max_pooling2d(inputs=rgb0, pool_size=[2, 2], strides=2)

            depth0 = tf.layers.conv2d(inputs=depth0,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
            depth0 = tf.layers.max_pooling2d(inputs=depth0, pool_size=[2, 2], strides=2)
            #
            '''
            rgb1 = tf.layers.conv2d(inputs=rgb1,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
            rgb1 = tf.layers.max_pooling2d(inputs=rgb1, pool_size=[2, 2], strides=2)
            rgb1 = tf.layers.conv2d(inputs=rgb1,filters=32,kernel_size=[4, 4],padding="same",activation=tf.nn.relu)
            rgb1 = tf.layers.max_pooling2d(inputs=rgb1, pool_size=[2, 2], strides=2)
            rgb1 = tf.layers.conv2d(inputs=rgb1,filters=32,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
            rgb1 = tf.layers.max_pooling2d(inputs=rgb1, pool_size=[2, 2], strides=2)
            rgb1 = tf.contrib.layers.flatten(rgb1)

            depth1 = tf.layers.conv2d(inputs=depth1,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
            depth1 = tf.layers.max_pooling2d(inputs=depth1, pool_size=[2, 2], strides=2)
            depth1 = tf.contrib.layers.flatten(depth1)
            '''
            #x = tf.concat([rgb0, depth0, rgb1, depth1], 1)
            rgb0 = tf.contrib.layers.flatten(rgb0)
            depth0 = tf.contrib.layers.flatten(depth0)
            x = tf.concat([rgb0, depth0, last_act, plane_box, bird_box, maybe_plane_box, maybe_bird_box], axis=1)
            # END NEW


            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        drop_out = True
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = obs
            (v, o, r, rgbd) = tf.split(out, [1, 3, 3, int(out.shape[1])-7], 1)
            rgbd = tf.reshape(rgbd, [-1, 288, 256, 4])
            #rgbd = tf.reshape(rgbd, [-1, 144, 256, 4])

            #(rgbd0, rgbd1) = tf.split(rgbd, [144, 144], 1)
            (rgb, depth) = tf.split(rgbd, [3,1], 3)
            #(rgb1, depth1) = tf.split(rgbd1, [3,1], 3)
            '''
            with tf.variable_scope("convnet"):
                for num_outputs, kernel_size, stride in convs:
                rgb0 = layers.convolution2d(rgb0,
                       num_outputs=num_outputs,
                       kernel_size=kernel_size,
                       stride=stride,
                       activation_fn=tf.nn.relu)
                rgb0 = tf.layers.batch_normalization(rgb0)
                for num_outputs, kernel_size, stride in convs:
                rgb1 = layers.convolution2d(rgb1,
                       num_outputs=num_outputs,
                       kernel_size=kernel_size,
                       stride=stride,
                       activation_fn=tf.nn.relu)
                rgb1 = tf.layers.batch_normalization(rgb1)
            '''
            '''
                for num_outputs, kernel_size, stride in convs:
                depth0 = layers.convolution2d(depth0,
                       num_outputs=num_outputs,
                       kernel_size=kernel_size,
                       stride=stride,
                       activation_fn=tf.nn.relu)
                depth0 = tf.layers.batch_normalization(depth0)
                for num_outputs, kernel_size, stride in convs:
                depth1 = layers.convolution2d(depth1,
                       num_outputs=num_outputs,
                       kernel_size=kernel_size,
                       stride=stride,
                       activation_fn=tf.nn.relu)
                depth1 = tf.layers.batch_normalization(depth1)
                '''
            rgb_out = tiny_yolo(rgb)
            #rgb1_out = tiny_yolo(rgb1)
            depth_out = tiny_yolo(depth)
            #depth1_out = tiny_yolo(depth1)

            #rgb0_out = layers.flatten(rgb0)
            #rgb1_out = layers.flatten(rgb1)
            #depth0_out = layers.flatten(depth0)
            #depth1_out = layers.flatten(depth1)
            #x = tf.concat([v, o, r, rgb0_out, depth0_out], 1)
            x = tf.concat([v, o, r, rgb_out, depth_out], axis=1)
            #conv_out = layers.flatten(out)
            x = tf.layers.dense(x, 2048)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 2048)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x


    def call_old(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            (last_act, plane_box, bird_box, maybe_plane_box, maybe_bird_box, rgbd) = tf.split(x, [2, 4, 4, 4, 4, int(x.shape[1])-(2+4+4+4+4)], 1)
            rgbd = tf.reshape(rgbd, [-1, 144, 256, 4])

            # NEW
            # For 2 Frames of RGB
            #(rgb0, rgb1) = tf.split(x, [144, 144], 1)
            (rgb0, depth0) = tf.split(rgbd, [3,1], 3)
            #(rgb1, depth1) = tf.split(rgb1, [3,1], 3)
            rgb0 = tf.layers.conv2d(inputs=rgb0,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
            rgb0 = tf.layers.max_pooling2d(inputs=rgb0, pool_size=[2, 2], strides=2)
            rgb0 = tf.layers.conv2d(inputs=rgb0,filters=32,kernel_size=[4, 4],padding="same",activation=tf.nn.relu)
            rgb0 = tf.layers.max_pooling2d(inputs=rgb0, pool_size=[2, 2], strides=2)
            rgb0 = tf.layers.conv2d(inputs=rgb0,filters=32,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
            rgb0 = tf.layers.max_pooling2d(inputs=rgb0, pool_size=[2, 2], strides=2)

            depth0 = tf.layers.conv2d(inputs=depth0,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
            depth0 = tf.layers.max_pooling2d(inputs=depth0, pool_size=[2, 2], strides=2)
            #
            '''
            rgb1 = tf.layers.conv2d(inputs=rgb1,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
            rgb1 = tf.layers.max_pooling2d(inputs=rgb1, pool_size=[2, 2], strides=2)
            rgb1 = tf.layers.conv2d(inputs=rgb1,filters=32,kernel_size=[4, 4],padding="same",activation=tf.nn.relu)
            rgb1 = tf.layers.max_pooling2d(inputs=rgb1, pool_size=[2, 2], strides=2)
            rgb1 = tf.layers.conv2d(inputs=rgb1,filters=32,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
            rgb1 = tf.layers.max_pooling2d(inputs=rgb1, pool_size=[2, 2], strides=2)
            rgb1 = tf.contrib.layers.flatten(rgb1)

            depth1 = tf.layers.conv2d(inputs=depth1,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
            depth1 = tf.layers.max_pooling2d(inputs=depth1, pool_size=[2, 2], strides=2)
            depth1 = tf.contrib.layers.flatten(depth1)
            '''
            #x = tf.concat([rgb0, depth0, rgb1, depth1], 1)
            rgb0 = tf.contrib.layers.flatten(rgb0)
            depth0 = tf.contrib.layers.flatten(depth0)
            x = tf.concat([rgb0, depth0, last_act, plane_box, bird_box, maybe_plane_box, maybe_bird_box], axis=1)
            # END NEW

            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
