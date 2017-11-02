import tensorflow as tf
import tensorflow.contrib.layers as layers


def _mlp(hiddens, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)
        q_out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return q_out


def mlp(hiddens=[], layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp(hiddens, layer_norm=layer_norm, *args, **kwargs)


def _cnn_to_mlp(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        conv_out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = conv_out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)
                action_out = tf.nn.relu(action_out)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = conv_out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        state_out = layers.layer_norm(state_out, center=True, scale=True)
                    state_out = tf.nn.relu(state_out)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return q_out

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
    conv0 = tf.layers.conv2d(inputs=X,filters=16,kernel_size=[3, 3],strides=1,padding="same",activation=tf.nn.relu)
    max1 = tf.layers.max_pooling2d(inputs=conv0, pool_size=[2, 2],strides=2)
    conv2 = tf.layers.conv2d(inputs=max1,filters=32,kernel_size=[3, 3],strides=1,padding="same",activation=tf.nn.relu)
    max3 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2, 2],strides=2)
    conv4 = tf.layers.conv2d(inputs=max3,filters=16,kernel_size=[1, 1],strides=1,padding="same",activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(inputs=conv4,filters=128,kernel_size=[3, 3],strides=1,padding="same",activation=tf.nn.relu)
    conv6 = tf.layers.conv2d(inputs=conv5,filters=16,kernel_size=[1, 1],strides=1,padding="same",activation=tf.nn.relu)
    conv7 = tf.layers.conv2d(inputs=conv6,filters=128,kernel_size=[3, 3],strides=1,padding="same",activation=tf.nn.relu)
    max8 = tf.layers.max_pooling2d(inputs=conv7,pool_size=[2, 2],strides=2)
    conv9 = tf.layers.conv2d(inputs=max8,filters=32,kernel_size=[1, 1],strides=1,padding="same",activation=tf.nn.relu)
    conv10 = tf.layers.conv2d(inputs=conv9,filters=256,kernel_size=[3, 3],strides=1,padding="same",activation=tf.nn.relu)
    conv11 = tf.layers.conv2d(inputs=conv10,filters=32,kernel_size=[1, 1],strides=1,padding="same",activation=tf.nn.relu)
    conv12 = tf.layers.conv2d(inputs=conv11,filters=256,kernel_size=[3, 3],strides=1,padding="same",activation=tf.nn.relu)
    max13 = tf.layers.max_pooling2d(inputs=conv12,pool_size=[2, 2],strides=2)
    conv14 = tf.layers.conv2d(inputs=max13,filters=64,kernel_size=[1, 1],strides=1,padding="same",activation=tf.nn.relu)
    conv15 = tf.layers.conv2d(inputs=conv14,filters=512,kernel_size=[3, 3],strides=1,padding="same",activation=tf.nn.relu)
    conv16 = tf.layers.conv2d(inputs=conv15,filters=64,kernel_size=[1, 1],strides=1,padding="same",activation=tf.nn.relu)
    conv17 = tf.layers.conv2d(inputs=conv16,filters=512,kernel_size=[3, 3],strides=1,padding="same",activation=tf.nn.relu)
    conv18 = tf.layers.conv2d(inputs=conv17,filters=128,kernel_size=[1, 1],strides=1,padding="same",activation=tf.nn.relu)
    conv19 = tf.layers.conv2d(inputs=conv18,filters=1000,kernel_size=[1, 1],strides=1,padding="same",activation=tf.nn.relu)
    avg20 = tf.layers.average_pooling2d(inputs=conv19,pool_size=[conv19.shape[1],conv19.shape[2]],strides=1)
    return layers.flatten(avg20)

def GOTURN_cnn(X):
    conv1 = tf.layers.conv2d(inputs=X,filters=96,kernel_size=[11, 11],strides=4,padding="same",activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3],strides=2)
    norm1 = tf.layers.batch_normalization(pool1)
    conv2 = tf.layers.conv2d(inputs=norm1,filters=256,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3],strides=2)
    norm2 = tf.layers.batch_normalization(pool2)
    conv3 = tf.layers.conv2d(inputs=norm2,filters=384,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv3,filters=384,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(inputs=conv4,filters=256,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3],strides=2)
    flat_pool5 = layers.flatten(pool5)
    '''
    conv1_p = tf.layers.conv2d(inputs=X,filters=96,kernel_size=[11, 11],strides=4,padding="same",activation=tf.nn.relu)
    pool1_p = tf.layers.max_pooling2d(inputs=conv1_p, pool_size=[3, 3],strides=2)
    norm1_p = tf.layers.batch_normalization(pool1_p)
    conv2_p = tf.layers.conv2d(inputs=norm1_p,filters=256,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
    pool2_p = tf.layers.max_pooling2d(inputs=conv2_p, pool_size=[3, 3],strides=2)
    norm2_p = tf.layers.batch_normalization(pool2_p)
    conv3_p = tf.layers.conv2d(inputs=norm2_p,filters=384,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    conv4_p = tf.layers.conv2d(inputs=conv3_p,filters=384,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    conv5_p = tf.layers.conv2d(inputs=conv4_p,filters=256,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
    pool5_p = tf.layers.max_pooling2d(inputs=conv5_p, pool_size=[3, 3],strides=2)
    flat_pool5_p = layers.flatten(pool5_p)
    '''

    #concat = tf.concat([flat_pool5, flat_pool5_p], axis=1)
    return flat_pool5

def GOTURN_forward(X):
    fc6 = layers.fully_connected(X, num_outputs=4096, activation_fn=tf.nn.relu)
    fc6 = tf.layers.dropout(fc6, rate=0.5)
    fc7 = layers.fully_connected(fc6, num_outputs=4096, activation_fn=tf.nn.relu)
    fc7 = tf.layers.dropout(fc7, rate=0.5)
    fc7b = layers.fully_connected(fc7, num_outputs=4096, activation_fn=tf.nn.relu)
    fc7b = tf.layers.dropout(fc7b, rate=0.5)
    fc8 = layers.fully_connected(fc7b, num_outputs=4, activation_fn=tf.nn.relu)
    return fc8

def _cnn_to_mlp_custom(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False, layer_norm=False):
    drop_out = True
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        (v, o, r, rgbd) = tf.split(out, [1, 3, 3, int(out.shape[1])-7], 1)
        rgbd = tf.reshape(rgbd, [-1, 288, 256, 4])
        #rgbd = tf.reshape(rgbd, [-1, 144, 256, 4])

        (rgbd0, rgbd1) = tf.split(rgbd, [144, 144], 1)
        (rgb0, depth0) = tf.split(rgbd0, [3,1], 3)
        (rgb1, depth1) = tf.split(rgbd1, [3,1], 3)
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
        rgb1_out = tiny_yolo(rgb1)
        depth0_out = tiny_yolo(depth0)
        depth1_out = tiny_yolo(depth1)

        #rgb0_out = layers.flatten(rgb0)
        #rgb1_out = layers.flatten(rgb1)
        #depth0_out = layers.flatten(depth0)
        #depth1_out = layers.flatten(depth1)
        conv_out = tf.concat([v, o, r, rgb0_out, depth0_out, rgb1_out, depth1_out], 1)
        #conv_out = tf.concat([rgb0_out, rgb1_out], axis=1)
        #conv_out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = conv_out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)
                if drop_out:
                    action_out = tf.layers.dropout(action_out, rate=0.5)
                action_out = tf.nn.relu(action_out)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = conv_out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        state_out = layers.layer_norm(state_out, center=True, scale=True)
                    if drop_out:
                        action_out = tf.layers.dropout(action_out, rate=0.5)
                    state_out = tf.nn.relu(state_out)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return q_out

def cnn_to_mlp_custom(convs, hiddens, dueling=False, layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_to_mlp_custom(convs, hiddens, dueling, layer_norm=layer_norm, *args, **kwargs)


def cnn_to_mlp(convs, hiddens, dueling=False, layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_to_mlp(convs, hiddens, dueling, layer_norm=layer_norm, *args, **kwargs)

