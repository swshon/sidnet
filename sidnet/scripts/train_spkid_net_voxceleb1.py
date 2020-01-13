import os,sys
import tensorflow as tf
import numpy as np
sys.path.insert(0, './scripts')
sys.path.insert(0, './models')
from tqdm import tqdm
import csv
from sklearn.metrics import roc_curve, auc

from tensorflow.contrib.learn.python.learn.datasets import base


def accuracy(predictions, labels):
    pred_class = np.argmax(predictions, 1)
    true_class = np.argmax(labels, 1)
    return (100.0 * np.sum(pred_class == true_class) / predictions.shape[0])


#### function for read tfrecords
def read_and_decode_emnet_mfcc_fixed(filename,win_len,dim):
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
    feats2d = tf.reshape(feats, [win_len, dim])
#     feats2d = tf.expand_dims(feats2d,-1)
    feats1d = feats
    return labels, shapes, feats2d


#### function for read tfrecords
def read_and_decode_emnet_mfcc(filename):
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
    feats2d = tf.reshape(feats.values, shapes)
#     feats2d = tf.expand_dims(feats2d,-1)
    feats1d = feats.values
    return labels, shapes, feats2d


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


### Variable Initialization


import argparse
parser = argparse.ArgumentParser(description="To train speaker verification network using voxceleb1", add_help=True)
parser.add_argument("--numgpus", type=int, default=1,help="source data")
parser.add_argument("--print_eer_interval", type=int, default=2000, help="target data")
parser.add_argument("--print_loss_interval", type=int, default=2000, help="target data")
parser.add_argument("--print_train_acc_interval", type=int, default=2000, help="target data")
parser.add_argument("--max_save_limit", type=int, default=500, help="target data")
parser.add_argument("--tfrecords_dir", type=str, default='./data/tfrecords', help="target data")
parser.add_argument("--save_dir", type=str, default='./saver', help="target data")
parser.add_argument("--save_interval", type=int, default=16000, help="target data")

parser.add_argument("--model_name", type=str, default='spk2vec_test32_mean_cmn_slim_ver', help="target data")
parser.add_argument("--learning_rate", type=float, default='0.005', help="target data")
parser.add_argument("--input_dim", type=int, default=40, help="target data")
parser.add_argument("--mini_batch", type=int, default=4, help="target data")
parser.add_argument("--feat_type", type=str, default='mfcc_win400_hop160_fixed298', help="target data")
parser.add_argument("--train_data", type=str, default='voxceleb1_dev', help="target data")
parser.add_argument("--train_total_split", type=int, default=100, help="target data")
parser.add_argument("--test_data", type=str, default='voxceleb1_test', help="target data")
parser.add_argument("--test_total_split", type=int, default=1, help="target data")
parser.add_argument("--softmax_num", type=int, default=1211, help="target data")
parser.add_argument("--resume_startpoint", type=int, default=0, help="0 if you don't have to resume")
parser.add_argument("--max_iteration", type=int, default=800000, help="target data")
parser.add_argument("--fixed_input_frame", type=int, default=298, help="target data")
parser.add_argument("--optimizer", type=str, default='adam', help="sgd,rms or adam")
args = parser.parse_known_args()[0]


NUMGPUS = args.numgpus
SAVE_INTERVAL = args.save_interval
LOSS_INTERVAL = args.print_loss_interval
MAX_SAVEFILE_LIMIT = args.max_save_limit
EERTEST_INTERVAL = args.print_eer_interval
TESTSET_INTERVAL = args.print_train_acc_interval
# duration = np.empty(0,dtype='int')
# spklab = []
TFRECORDS_FOLDER = args.tfrecords_dir
SAVER_FOLDERNAME = args.save_dir

NN_MODEL = args.model_name
LEARNING_RATE = args.learning_rate
INPUT_DIM = args.input_dim
# IS_BATCHNORM = 'True'
BATCHSIZE = args.mini_batch
FEAT_TYPE = args.feat_type
DATA_NAME = args.train_data
TOTAL_SPLIT = args.train_total_split
SOFTMAX_NUM = args.softmax_num
RESUME_STARTPOINT = args.resume_startpoint
MAX_ITER = args.max_iteration
TEST_SET_NAME = args.test_data
INPUT_LENGTH = args.fixed_input_frame # in frame

resume = False
# is_batchnorm = False
if RESUME_STARTPOINT > 0:
    resume = True

SAVER_FOLDERNAME = SAVER_FOLDERNAME+'/' + NN_MODEL+'_'+str(INPUT_LENGTH)+'frame_'+FEAT_TYPE
# if IS_BATCHNORM=='True':
#     SAVER_FOLDERNAME = SAVER_FOLDERNAME + '_BN'
#     is_batchnorm = True
nn_model = __import__(NN_MODEL)

records_shuffle_list = []
for i in range(1,TOTAL_SPLIT+1):
    records_shuffle_list.append(TFRECORDS_FOLDER+'/'+DATA_NAME+'_'+FEAT_TYPE+'.'+str(i)+'.tfrecords')


labels,shapes,feats = read_and_decode_emnet_mfcc_fixed(records_shuffle_list,int(INPUT_LENGTH),int(INPUT_DIM))
labels_batch,feats_batch,shapes_batch = tf.train.shuffle_batch(
    [labels, feats,shapes], batch_size=BATCHSIZE, allow_smaller_final_batch=False,
    capacity=50000,num_threads=4,min_after_dequeue=10000)



FEAT_TYPE = FEAT_TYPE.split('_exshort')[0]
FEAT_TYPE = FEAT_TYPE.split('_fixed')[0]
records_test_list = []
for i in range(1,2):
    records_test_list.append(TFRECORDS_FOLDER+'/'+TEST_SET_NAME+'_'+FEAT_TYPE+'.'+str(i)+'.tfrecords')


#data for validation
vali_labels,vali_shapes,vali_feats = read_and_decode_emnet_mfcc(records_test_list)

# test trials
tst_segments = []
tst_trials=[]
tst_enrolls = []
tst_tests = []

with open('/data/sls/scratch/swshon/dataset/voxceleb1/voxceleb1_test.txt','rb') as csvfile:
    row = csv.reader(csvfile, delimiter=' ')
    for line in row:
        tst_trials = np.append(tst_trials,line[0])
        tst_segments.extend([line[1]])
        tst_segments.extend([line[2]])
        tst_enrolls.extend([line[1]])
        tst_tests.extend([line[2]])

[tst_segments,tst_spklabel] = np.unique(tst_segments, return_inverse=True)

tst_dict=dict()
tst_list=[]
for value,key in enumerate(tst_segments):
    tst_dict[key]=value

tst_enrolls_idx = []
tst_tests_idx = []
for index in range(len(tst_enrolls)):
    tst_enrolls_idx = np.append(tst_enrolls_idx, int(tst_dict[tst_enrolls[index]]))
    tst_tests_idx = np.append(tst_tests_idx, int(tst_dict[tst_tests[index]]))

print np.shape(tst_list)
print np.shape(tst_trials), np.shape(tst_enrolls), np.shape(tst_enrolls_idx)


### Initialize network related variables

# from tensorflow.contrib.framework.python.ops import arg_scope
# from tensorflow.contrib.slim.python.slim.nets import resnet_utils
# from tensorflow.contrib.slim.python.slim.nets import resnet_v2
# from tensorflow.contrib import layers as layers_lib
# from tensorflow.contrib.layers.python.layers import layers
# from tensorflow.contrib.layers.python.layers import utils


with tf.device('/cpu:0'):

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step,
                                               50000, 0.98, staircase=True)

    if args.optimizer=='adam':
        opt = tf.train.AdamOptimizer(LEARNING_RATE)
    elif args.optimizer=='rms':
        opt = tf.train.RMSPropOptimizer(LEARNING_RATE)
    elif args.optimizer=='sgd':
        opt = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        print "wrong optimizer"
        quit()


    emnet_losses = []
    emnet_grads = []

#     test_feat_batch = tf.placeholder(tf.float32, [None,None,np.int(INPUT_DIM),1],name="test_feat_batch") #for 2d-CNN
    test_feat_batch = tf.placeholder(tf.float32, [None,None,np.int(INPUT_DIM)],name="test_feat_batch")
    test_label_batch = tf.placeholder(tf.int32, [None],name="test_label_batch")
    test_shape_batch = tf.placeholder(tf.int32, [None,2],name="test_shape_batch")
    

#     #define resnet structure
#     blocks = [
#         resnet_v2.resnet_v2_block('block1', base_depth=16, num_units=3, stride=2),
#         resnet_v2.resnet_v2_block('block2', base_depth=32, num_units=4, stride=2),
#         resnet_v2.resnet_v2_block('block3', base_depth=64, num_units=6, stride=2),
#         resnet_v2.resnet_v2_block('block4', base_depth=128, num_units=3, stride=1),
#     ]



    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(NUMGPUS):
            with tf.device('/gpu:%d' % i):
#                 emnet_loss,emnet = resnet_v2_spkid(feats_batch,
#                                               labels_batch,
#                                               num_classes = SOFTMAX_NUM,
#                                               is_training = True,
#                                               global_pool = True,
#                                               output_stride = None,
#                                               reuse = False,
#                                               scope = 'resnet_v2_softmax')

                emnet = nn_model.nn(feats_batch, labels_batch,labels_batch, shapes_batch, SOFTMAX_NUM,True,INPUT_DIM)
                tf.get_variable_scope().reuse_variables()
                grads = opt.compute_gradients(emnet.loss)
                emnet_losses.append(emnet.loss)
                emnet_grads.append(grads)

        with tf.device('/gpu:0'):
            emnet_validation = nn_model.nn(test_feat_batch,test_label_batch,test_label_batch,test_shape_batch, SOFTMAX_NUM,False,INPUT_DIM);
#             _,emnet_validation = resnet_v2_spkid(test_feat_batch,
#                                           test_label_batch,
#                                           num_classes = SOFTMAX_NUM,
#                                           is_training = False,
#                                           global_pool = True,
#                                           output_stride = None,
#                                           reuse = True,
#                                           scope = 'resnet_v2_softmax')
            tf.get_variable_scope().reuse_variables()

    loss = tf.reduce_mean(emnet_losses)
    grads = average_gradients(emnet_grads)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)    
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver(max_to_keep=MAX_SAVEFILE_LIMIT)
#     verinet_vars = tf.get_collection(tf.GraphKeys.VARIABLES,scope="verinet")
#     pretrain_saver = tf.train.Saver(var_list = verinet_vars)

    tf.initialize_all_variables().run()
    tf.train.start_queue_runners(sess=sess)

    #load entire test set
    test_list = np.loadtxt('data/voxceleb1_test/wav.scp',dtype='string',usecols=[1])
    test_feats =[]
    test_labels = []
    test_shapes = []
    for iter in range(len(test_list)):
#     for iter in range(10):
        a,b,c = sess.run([vali_feats,vali_labels,vali_shapes])
        test_feats.extend([a])
        test_labels.extend([b])
        test_shapes.extend([c])

    ### Training neural network
    if resume:
        saver.restore(sess,SAVER_FOLDERNAME+'/model'+str(RESUME_STARTPOINT)+'.ckpt-'+str(RESUME_STARTPOINT))
        
        
    for step in range(RESUME_STARTPOINT,MAX_ITER):

        _, _,loss_v,mean_loss = sess.run([apply_gradient_op,update_ops, emnet.loss,loss])


        if np.isnan(loss_v):
            print ('Model diverged with loss = NAN')
            quit()

        if step % EERTEST_INTERVAL ==0 and step>=RESUME_STARTPOINT:
            embeddings = []

            for iter in range(len(test_list)):
                eb = emnet_validation.eb.eval({test_feat_batch:[test_feats[iter]], test_label_batch:[test_labels[iter]], test_shape_batch:[test_shapes[iter]]})
#                 eb = emnet_validation['resnet_v2_softmax/fc2'].eval({test_feat_batch:[test_feats[iter]], test_label_batch:[test_labels[iter]], test_shape_batch:[test_shapes[iter]]})
                embeddings.extend([eb])
            embeddings = np.squeeze(embeddings)

#             norms = np.linalg.norm(embeddings,axis=1)
#             for iter, norm in enumerate(norms):
#                 embeddings[iter,:] = embeddings[iter,:]/norm
    

            scores = np.zeros([len(tst_enrolls_idx)])
            for iter in range(len(tst_enrolls_idx)):
                scores[iter] =  embeddings[int(tst_enrolls_idx[iter]),:].dot( embeddings[int(tst_tests_idx[iter]),:].transpose())

            tst_trials = map(int,tst_trials)
            fpr,tpr,threshold = roc_curve(tst_trials,scores,pos_label=1)
            fnr = 1-tpr
            EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
            EER = fpr[np.argmin(np.absolute((fnr-fpr)))]
            print ('Step %d: loss %.3f, lr : %.5f, EER : %f' % (step,mean_loss, sess.run(learning_rate),EER))


        if step % TESTSET_INTERVAL ==0 and step >=RESUME_STARTPOINT:
            prediction = np.empty((0,5))
            _label = np.int64([])
            a,b,c = sess.run([feats_batch,labels_batch,shapes_batch])
            prediction, _label= sess.run([emnet_validation.o1, emnet_validation.label ], feed_dict={
                test_feat_batch:a,
                test_label_batch:b,
                test_shape_batch:c
            })
            spklab_num_mat = np.eye(SOFTMAX_NUM)[_label]
            acc = accuracy(prediction, spklab_num_mat)
            print ('Step %d: loss %.3f, lr : %.5f, Accuracy : %f' % (step,mean_loss, sess.run(learning_rate),acc))

        if step % LOSS_INTERVAL ==0:
            print ('Step %d: loss %.3f, lr : %.5f' % (step, mean_loss, sess.run(learning_rate)))

        if step % SAVE_INTERVAL == 0 and step >=RESUME_STARTPOINT:
            saver.save(sess, SAVER_FOLDERNAME+'/model'+str(step)+'.ckpt',global_step=step)        
