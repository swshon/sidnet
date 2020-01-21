import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib import slim as contrib_slim

slim = contrib_slim

class nn:
    # Create model
    def __init__(self, x1, y_, num_classes, is_training, global_pool, output_stride, reuse, scope):
        #define resnet structure
        blocks = [
            resnet_v2.resnet_v2_block('block1', base_depth=16, num_units=3, stride=[1, 1]),
            resnet_v2.resnet_v2_block('block2', base_depth=32, num_units=4, stride=[1, 2]),
            resnet_v2.resnet_v2_block('block3', base_depth=64, num_units=6, stride=[1, 2]),
            resnet_v2.resnet_v2_block('block4', base_depth=128, num_units=3, stride=[1, 2]),
        ]
        inputlayer = self.cmn(x1)
        loss,end_points = self.resnet_v2_spkid(inputlayer,
                               y_,
                               blocks,
                               num_classes = num_classes,
                               is_training = is_training,
                               global_pool = global_pool,
                               output_stride = output_stride,
                               reuse = reuse,
                               scope = scope)

        self.end_points = end_points
        self.loss = loss
        self.label=y_

    def resnet_v2_spkid(self,inputs,
                        spk_labels,
                        blocks,
                        num_classes,
                        is_training,
                        global_pool,
                        output_stride,
                        reuse,
                        scope):


        with arg_scope(resnet_v2.resnet_arg_scope()):
            with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
                end_points_collection = sc.original_name_scope + '_end_points'
                with arg_scope([layers_lib.conv2d, resnet_v2.bottleneck, slim.conv2d,
                                     self.stack_blocks_dense], outputs_collections=end_points_collection):
                    with arg_scope([layers_lib.conv2d], 
                            weights_regularizer = None,
                            weights_initializer = tf.contrib.layers.xavier_initializer(),
                            biases_initializer= tf.constant_initializer(0.001) ):

                        with arg_scope([layers.batch_norm], is_training=is_training,
                                decay=0.9, epsilon=1e-3, scale=True,
                                param_initializers={
                                    "beta": tf.constant_initializer(value=0),
                                    "gamma": tf.random_normal_initializer(mean=1, stddev=0.045),
                                    "moving_mean": tf.constant_initializer(value=0),
                                    "moving_variance": tf.constant_initializer(value=1)} ):
                            net = inputs
                            with arg_scope([layers_lib.conv2d], activation_fn=None, normalizer_fn=None,weights_regularizer=None):
                                net = resnet_utils.conv2d_same(net, 64, 13, 1, scope='conv1')
                            # net = layers.max_pool2d(net, [2, 2], stride=2, scope='pool1')
                            # net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
                            net = self.stack_blocks_dense(net, blocks, output_stride)
                            net = layers.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                            end_points = utils.convert_collection_to_dict(end_points_collection)
                            net = layers_lib.conv2d(net, 512, [1, 5], stride=1, activation_fn=None,
                                            normalizer_fn=None, scope='res_fc',padding='VALID')
                            end_points[sc.name + '/res_fc'] = net
                            net = layers.batch_norm(net, activation_fn=tf.nn.relu, scope='res_fc_bn')

                            if global_pool:
                                # net : batchsize X W(frame_length) X 1 X Dim
                                # Global average pooling.
#                                net = tf.reduce_mean(net, [1], name='pool5', keep_dims=True)

                                # Apply attention
#                                 attention =self.attention_layer(net)
#                                 end_points['attention']=attention
#                                 net = tf.multiply(net, attention) # weighted activations
#                                 net = tf.reduce_sum(net,1, name='pool_att_mean', keep_dims=True)
#                                 end_points['global_pool'] = net

                        
                                # Apply attention + stats
                                attention =self.attention_layer(net)
                                end_points['attention']=attention

                                #std = tf.multiply(tf.multiply(net,net), attention)
                                #std = tf.reduce_sum(std,1, keep_dims=True)

                                #net = tf.multiply(net, attention) # weighted activations
                                #mean = tf.reduce_sum(net,1, name='pool_att_mean', keep_dims=True)
                                #std = tf.sqrt(std - mean**2)
                                #net = tf.concat([mean,std],3)
                                
#                                 mean = tf.multiply(net, attention) # weighted activation
#                                 mean = tf.reduce_sum(mean,1, name='pool_att_mean', keep_dims=True)
#                                 std = tf.reduce_sum(tf.sqrt(tf.multiply(tf.square(net-mean),attention)),1,name='pool_att_std',keep_dims=True)
                                mean,std=tf.nn.weighted_moments(net,1,attention,keep_dims=True)
                                net = tf.concat([mean,std],3)
                                end_points['global_pool'] = net
                        
                                # Global statistical pooling                            
#                                 mean,var = tf.nn.moments(net,1,name='pool5', keep_dims=True)
#                                 net = tf.concat([mean,var],3)
#                                 end_points['global_pool'] = net

                            #Fully Connected layers
                            #fc1
                            net = layers_lib.conv2d(net, 1000, [1, 1], stride=1, activation_fn=None,
                                            normalizer_fn=None, scope='fc1')
                            end_points[sc.name + '/fc1'] = net
                            net = layers.batch_norm(net, activation_fn=tf.nn.relu, scope='fc1_bn')
                            #fc2
                            net = layers_lib.conv2d(net, 512, [1, 1], stride=1, activation_fn=None,
                                            normalizer_fn=None, scope='fc2')
                            end_points[sc.name + '/fc2'] = net

                            ## output layer
                            # For AM-softmax loss
                            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                            end_points[sc.name + '/spatial_squeeze'] = net
                            net, embedding = self.AM_logits_compute(net, spk_labels, num_classes,is_training)
                            end_points[sc.name + '/logits'] = net
                            end_points[sc.name + '/fc3'] = embedding

                            # for softmax 
#                             net = layers.batch_norm(net, activation_fn=tf.nn.relu, scope='fc2_bn')
#                             net = layers_lib.conv2d(net, num_classes, [1, 1], stride=1, activation_fn=None,
#                                             normalizer_fn=None, scope='logits')
#                             end_points[sc.name + '/logits'] = net
#                             net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
#                             end_points[sc.name + '/spatial_squeeze'] = net


                            #loss
                            end_points['predictions'] = layers.softmax(net, scope='predictions')
                            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=spk_labels, logits=net))
                            end_points[sc.name + '/loss'] = loss
                            end_points[sc.name + '/spk_labels'] = spk_labels

                            return loss, end_points
                    
                    



    def cmvn(self,temp_batch):
        mean = tf.reduce_mean(temp_batch,1,keep_dims=True)
        std = tf.sqrt( tf.reduce_mean( tf.square(temp_batch-mean),1,keep_dims=True) )
        temp_batch = tf.divide(tf.subtract(temp_batch , mean),std)
        return temp_batch

    def cmn(self,temp_batch):
        temp_batch = tf.squeeze(temp_batch,axis=-1)
        mean = tf.reduce_mean(temp_batch,1,keep_dims=True)
        std = tf.sqrt( tf.reduce_mean( tf.square(temp_batch-mean),1,keep_dims=True) )
        temp_batch = tf.subtract(temp_batch , mean)
        temp_batch = tf.expand_dims(temp_batch,-1)
        return temp_batch

    @slim.add_arg_scope
    def stack_blocks_dense(self, net, blocks, output_stride=None,
                           store_non_strided_activations=False,
                           outputs_collections=None,
                           weights_regularizer = None,
                           weights_initializer = None,
                           biases_initializer = None):
      """Stacks ResNet `Blocks` and controls output feature density.
      First, this function creates scopes for the ResNet in the form of
      'block_name/unit_1', 'block_name/unit_2', etc.
      Second, this function allows the user to explicitly control the ResNet
      output_stride, which is the ratio of the input to output spatial resolution.
      This is useful for dense prediction tasks such as semantic segmentation or
      object detection.
      Most ResNets consist of 4 ResNet blocks and subsample the activations by a
      factor of 2 when transitioning between consecutive ResNet blocks. This results
      to a nominal ResNet output_stride equal to 8. If we set the output_stride to
      half the nominal network stride (e.g., output_stride=4), then we compute
      responses twice.
      Control of the output feature density is implemented by atrous convolution.
      Args:
        net: A `Tensor` of size [batch, height, width, channels].
        blocks: A list of length equal to the number of ResNet `Blocks`. Each
          element is a ResNet `Block` object describing the units in the `Block`.
        output_stride: If `None`, then the output will be computed at the nominal
          network stride. If output_stride is not `None`, it specifies the requested
          ratio of input to output spatial resolution, which needs to be equal to
          the product of unit strides from the start up to some level of the ResNet.
          For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,
          then valid values for the output_stride are 1, 2, 6, 24 or None (which
          is equivalent to output_stride=24).
        store_non_strided_activations: If True, we compute non-strided (undecimated)
          activations at the last unit of each block and store them in the
          `outputs_collections` before subsampling them. This gives us access to
          higher resolution intermediate activations which are useful in some
          dense prediction problems but increases 4x the computation and memory cost
          at the last unit of each block.
        outputs_collections: Collection to add the ResNet block outputs.
      Returns:
        net: Output tensor with stride equal to the specified output_stride.
      Raises:
        ValueError: If the target output_stride is not valid.
      """
      # The current_stride variable keeps track of the effective stride of the
      # activations. This allows us to invoke atrous convolution whenever applying
      # the next residual unit would result in the activations having stride larger
      # than the target output_stride.
      current_stride = 1

      # The atrous convolution rate parameter.
      rate = 1

      for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
          block_stride = 1
          for i, unit in enumerate(block.args):
            if store_non_strided_activations and i == len(block.args) - 1:
              # Move stride from the block's last unit to the end of the block.
              block_stride = unit.get('stride', 1)
              unit = dict(unit, stride=1)

            with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
              # If we have reached the target output_stride, then we need to employ
              # atrous convolution with stride=1 and multiply the atrous rate by the
              # current unit's stride for use in subsequent layers.
              if output_stride is not None and current_stride == output_stride:
                net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
                rate *= unit.get('stride', 1)

              else:
                net = block.unit_fn(net, rate=1, **unit)
                if isinstance(unit.get('stride', 1), int):
                    st = unit.get('stride', 1)
                else:
                    st = unit.get('stride', 1)[0]
                current_stride *= st
                if output_stride is not None and current_stride > output_stride:
                  raise ValueError('The target output_stride cannot be reached.')

          # Collect activations at the block's end before performing subsampling.
          net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

          # Subsampling of the block's output activations.
          if output_stride is not None and current_stride == output_stride:
            rate *= block_stride
          else:
            net = resnet_utils.subsample(net, block_stride)
            current_stride *= block_stride
            if output_stride is not None and current_stride > output_stride:
              raise ValueError('The target output_stride cannot be reached.')

      if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

      return net


    def AM_logits_compute(self,embeddings, label_batch, nrof_classes,is_training):
	# refer https://github.com/Joker316701882/Additive-Margin-Softmax.git
        m = 0.35
        s = 30
        with tf.variable_scope('AM_logits'):
            am_fc_norm = tf.nn.l2_normalize(embeddings, 1, 1e-10, name='embeddings')
            kernel = tf.get_variable(name='am_kernel',dtype=tf.float32,shape=[512,nrof_classes],initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-10, name='kernel_norm')
            cos_theta = tf.matmul(am_fc_norm, kernel_norm)
            cos_theta = tf.clip_by_value(cos_theta, -1,1) # for numerical steady
            phi = cos_theta - m
            label_onehot = tf.one_hot(label_batch, nrof_classes)
            adjust_theta = s * tf.where(tf.equal(label_onehot,1), phi, cos_theta)
            return adjust_theta,am_fc_norm
       


    def attention_layer(self, net):
        # net : batchsize X W(frame_length) X 1 X Dim
        channel_dim = net.get_shape()[3].value
        attention_dim = 64
        heads_dim = 1
        W = tf.get_variable(name='att_W',dtype=tf.float32, shape=[channel_dim,attention_dim],initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        #b = tf.get_variable(name='att_b',dtype=tf.float32, shape=[attention_dim],initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        v = tf.get_variable(name='att_v',dtype=tf.float32, shape=[attention_dim,heads_dim],initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        #k = tf.get_variable(name='att_k',dtype=tf.float32, shape=[heads_dim],initializer=tf.contrib.layers.xavier_initializer(uniform=True))

#         A = tf.nn.softmax( tf.matmul(tf.nn.relu(tf.matmul(net,W)), v), name='alphas',axis=1)
        A = tf.nn.softmax( tf.matmul(tf.nn.tanh(tf.matmul(net,W)), v), name='alphas',axis=1)
#        A = tf.nn.softmax( tf.bias_add(tf.matmul(tf.nn.relu( tf.nn.bias_add(tf.matmul(net,W),b) ), v), k), name='alphas',axis=1)
        # A: batchsize X W)(frame_length) X 1 X heads_dim
        return A


    def fc_layer(self, bottom, n_weight, name):
        print( bottom.get_shape())
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]

        initer = self.xavier_init(int(n_prev_weight),n_weight)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.random_uniform([n_weight],-0.001,0.001, dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc


    def batch_norm_wrapper_fc(self, inputs, is_training, name, is_batchnorm, decay = 0.999 ):
	if is_batchnorm:
            epsilon = 1e-3
            scale = tf.get_variable(name+'scale',dtype=tf.float32,initializer=tf.ones([inputs.get_shape()[-1]]) )
            beta = tf.get_variable(name+'beta',dtype=tf.float32,initializer= tf.zeros([inputs.get_shape()[-1]]) )
            pop_mean = tf.get_variable(name+'pop_mean',dtype=tf.float32,initializer = tf.zeros([inputs.get_shape()[-1]]), trainable=False)
            pop_var = tf.get_variable(name+'pop_var',dtype=tf.float32,initializer = tf.ones([inputs.get_shape()[-1]]), trainable=False)
            if is_training:
                batch_mean, batch_var = tf.nn.moments(inputs,[0])
                train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, epsilon)
            else:
                return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)
	else:
	    return inputs

    def xavier_init(self,n_inputs, n_outputs, uniform=True):
      if uniform:
        init_range = np.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
      else:
        stddev = np.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)
