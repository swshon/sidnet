import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import utils


class nn:
    # Create model
    def __init__(self, x1, y_, num_classes, is_training, global_pool, output_stride, reuse, scope):
        inputlayer = self.cmn(x1)
        end_points = self.net(inputlayer,
                               num_classes = num_classes,
                               is_training = is_training,
                               reuse = reuse,
                               scope = scope)

        self.loss    = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=end_points['predictions']),name="loss_mean")

        self.end_points = end_points
        self.label=y_


    def cmvn(self,temp_batch):
        mean = tf.reduce_mean(temp_batch,1,keep_dims=True)
        std = tf.sqrt( tf.reduce_mean( tf.square(temp_batch-mean),1,keep_dims=True) )
        temp_batch = tf.divide(tf.subtract(temp_batch , mean),std)
        return temp_batch

    def cmn(self,temp_batch):
        mean = tf.reduce_mean(temp_batch,1,keep_dims=True)
        std = tf.sqrt( tf.reduce_mean( tf.square(temp_batch-mean),1,keep_dims=True) )
        temp_batch = tf.subtract(temp_batch , mean)
        return temp_batch

    def net(self, inputs, num_classes, is_training, reuse, scope):

        with tf.variable_scope(scope, 'cnn_v1', [inputs], reuse=reuse) as sc:

            with arg_scope([layers.batch_norm], is_training=is_training,
                           decay=0.9, epsilon=1e-3,scale=True,
                           param_initializers={
                               "beta": tf.constant_initializer(value=0),
                               "gamma": tf.random_normal_initializer(mean=1, stddev=0.045),
                               'moving_mean':tf.constant_initializer(value=0),
                               'moving_variance':tf.constant_initializer(value=1)} ):

                with arg_scope([layers_lib.conv1d, layers_lib.fully_connected], activation_fn=None, normalizer_fn=None,
                               weights_regularizer = None,
                               weights_initializer = tf.contrib.layers.xavier_initializer(),
                               biases_initializer = tf.constant_initializer(0.001) ):

                    end_points={}

                    conv1 = layers_lib.conv1d(inputs, 1000, [5], stride=1, padding='SAME',scope='conv1')
                    conv1r = layers.batch_norm(conv1, activation_fn=tf.nn.relu, scope='bn1')

                    conv2 = layers_lib.conv1d(conv1r, 1000, [7], stride=2, padding='SAME',scope='conv2')
                    conv2r = layers.batch_norm(conv2, activation_fn=tf.nn.relu, scope='bn2')

                    conv3 = layers_lib.conv1d(conv2r, 1000, [1], stride=1, padding='SAME',scope='conv3')
                    conv3r = layers.batch_norm(conv3, activation_fn=tf.nn.relu, scope='bn3')

                    conv4 = layers_lib.conv1d(conv3r, 1500, [1], stride=1, padding='SAME',scope='conv4')
                    conv4r = layers.batch_norm(conv4, activation_fn=tf.nn.relu, scope='bn4')

                    mean = tf.reduce_mean(conv4r,1,keep_dims=True)
                    res1=tf.squeeze(mean,axis=1)


                    fc1 = layers_lib.fully_connected(res1,1500,scope='fc1')
    #                 fc1_bn = layers.batch_norm(fc1, activation_fn=tf.nn.relu, scope='bn5')
                    end_points[sc.name + '/fc1'] = fc1
                    fc1_bn = layers.batch_norm(fc1, activation_fn=None, scope='bn5')

                    fc2 = layers_lib.fully_connected(fc1_bn,600,scope='fc2')
                    end_points[sc.name + '/fc2'] = fc2
                    fc2_bn = layers.batch_norm(fc2, activation_fn=tf.nn.relu, scope='bn6')
                    fc3 = layers_lib.fully_connected(fc2_bn,num_classes,scope='fc3')
                    end_points['predictions'] = fc3

        return end_points
