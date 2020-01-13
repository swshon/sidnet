import os,sys
import tensorflow as tf
import numpy as np
sys.path.insert(0, './scripts')
sys.path.insert(0, './models')

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def accuracy(predictions, labels):
    pred_class = np.argmax(predictions, 1)
    true_class = np.argmax(labels, 1)
    return (100.0 * np.sum(pred_class == true_class) / predictions.shape[0])


#### function for read tfrecords
def read_and_decode_tfrecords_fixed(filename,win_len,dim,cnn1d=False):
    # first construct a queue containing a list of filenames.
    # this lets a user split up there dataset in multiple files to keep
    # size down
    filename_queue = tf.train.string_input_producer(filename, name = 'queue')
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'labels': tf.io.FixedLenFeature([], tf.int64),
            'shapes': tf.io.FixedLenFeature([2], tf.int64),
#             'features': tf.VarLenFeature( tf.float32)
            'features': tf.io.FixedLenFeature([win_len*dim], tf.float32)
        })
    # now return the converted data
    labels = features['labels']
    shapes = features['shapes']
    feats = features['features']
#     shapes = tf.cast(shapes, tf.int32)
    feats1dcnn = tf.reshape(feats, [win_len, dim])
    feats2dcnn = tf.expand_dims(feats1dcnn,-1)
#     feats1d = feats
    if cnn1d == True:
        return labels, shapes, feats1dcnn
    else :
        return labels, shapes, feats2dcnn


#### function for read tfrecords
def read_and_decode_tfrecords_variable(filename,cnn1d=False):
    # first construct a queue containing a list of filenames.
    # this lets a user split up there dataset in multiple files to keep
    # size down
    filename_queue = tf.train.string_input_producer(filename, name = 'queue')
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'labels': tf.io.FixedLenFeature([], tf.int64),
            'shapes': tf.io.FixedLenFeature([2], tf.int64),
            'features': tf.io.VarLenFeature( tf.float32)
        })
    # now return the converted data
    labels = features['labels']
    shapes = features['shapes']
    feats = features['features']
    shapes = tf.cast(shapes, tf.int32)
    feats1dcnn = tf.reshape(feats.values, shapes)
    feats2dcnn = tf.expand_dims(feats1dcnn,-1)
#     feats1d = feats.values
    if cnn1d == True:
        return labels, shapes, feats1dcnn
    else :
        return labels, shapes, feats2dcnn

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
