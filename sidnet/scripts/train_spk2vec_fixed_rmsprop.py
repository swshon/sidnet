import os,sys
import tensorflow as tf
import numpy as np
sys.path.insert(0, './scripts')
sys.path.insert(0, './models')
from tqdm import tqdm
import csv
from sklearn.metrics import roc_curve, auc

from tensorflow.contrib.learn.python.learn.datasets import base
#import nn_model as nn_model
#import nn_model as nn_model_foreval


def accuracy(predictions, labels):
    pred_class = np.argmax(predictions, 1)
    true_class = np.argmax(labels, 1)
#     print pred_class
#     print true_class
    return (100.0 * np.sum(pred_class == true_class) / predictions.shape[0])

def txtwrite(filename, dict):
    with open(filename, "w") as text_file:
        for key, vec in dict.iteritems():
            text_file.write('%s [' % key)
            for i, ele in enumerate(vec):
                text_file.write(' %f' % ele)
            text_file.write(' ]\n')

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)



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
NUMGPUS = 1
# BATCHSIZE = 4
# ITERATION = 2000000
SAVE_INTERVAL = 16000
LOSS_INTERVAL = 100
TESTSET_INTERVAL = 2000
MAX_SAVEFILE_LIMIT = 500
EERTEST_INTERVAL = 16000
# DATASET_NAME = 'post_pooled/train_gmm.h5'
# DURATION_LIMIT = 1000 #(utterance below DURATION_LIMIT/100 seconds will be mismissed :default=1000)
# SPKCOUNT_LIMIT = 3 #(speaker with equal or less than this number will be dismissed :default=3)
# MIXTURE = 2048  # = number of softmax layer
# filelist = ['../post_pooled/swb_gmm','../post_pooled/sre_gmm'] # dataset for training
# post_mean = np.empty((0,MIXTURE),dtype='float32')
# post_std = np.empty((0,MIXTURE),dtype='float32')
utt_label = []
duration = np.empty(0,dtype='int')
spklab = []
TFRECORDS_FOLDER = './data/tfrecords/'
# TFRECORDS_FOLDER = '/data/sls/scratch/swshon/exp_scratch/enhance_spkid_voxceleb1/data/tfrecords/'
SAVER_FOLDERNAME = 'saver'

if len(sys.argv)< 8:
    print "not enough arguments"
    print "command : ./new_training.py [nn_model_name] [learning rate] [input_dim(feat dim)] [is_batch_norm] [feature_filename]"
    print "(example) ./new_training.py nn_model 0.001 40 True aug_mfcc_fft512_hop160_vad_cmn"
resume = False
is_batchnorm = False
NN_MODEL = sys.argv[1]
LEARNING_RATE = np.float(sys.argv[2])
INPUT_DIM = sys.argv[3]
IS_BATCHNORM = sys.argv[4]
BATCHSIZE = int(sys.argv[5])
FEAT_TYPE = sys.argv[6]
DATA_NAME = sys.argv[7]
TOTAL_SPLIT = np.int(sys.argv[8])
SOFTMAX_NUM = np.int(sys.argv[9])
RESUME_STARTPOINT = np.int(sys.argv[10])
MAX_ITER = np.int(sys.argv[11])
TEST_SET_NAME = sys.argv[12]
INPUT_LENGTH = np.int(sys.argv[13]) # in frame
if RESUME_STARTPOINT > 0:
    resume = True

# NN_MODEL = 'new_nn_model_3sec'
# LEARNING_RATE = 0.005
# INPUT_DIM = 40
# IS_BATCHNORM = 'True'
# FEAT_TYPE = 'mfcc_fft512_hop160_vad_cmn'
# DATA_NAME = 'voxceleb1_dev_shuffle'
# TOTAL_SPLIT = np.int(100)
# SOFTMAX_NUM = np.int(1211)



SAVER_FOLDERNAME = 'saver/'+NN_MODEL+'_'+str(INPUT_LENGTH)+'frame_'+FEAT_TYPE
if IS_BATCHNORM=='True':
    SAVER_FOLDERNAME = SAVER_FOLDERNAME + '_BN'
    is_batchnorm = True
nn_model = __import__(NN_MODEL)


records_shuffle_list = []
for i in range(1,TOTAL_SPLIT+1):
    records_shuffle_list.append(TFRECORDS_FOLDER+DATA_NAME+'_'+FEAT_TYPE+'.'+str(i)+'.tfrecords')


labels,shapes,feats = read_and_decode_emnet_mfcc_fixed(records_shuffle_list,int(INPUT_LENGTH),int(INPUT_DIM))
# labels_batch,feats_batch,shapes_batch = tf.train.batch(
#     [labels, feats,shapes], batch_size=BATCHSIZE, dynamic_pad=True, allow_smaller_final_batch=True,
#     capacity=50)
labels_batch,feats_batch,shapes_batch = tf.train.shuffle_batch(
    [labels, feats,shapes], batch_size=BATCHSIZE, allow_smaller_final_batch=False,
    capacity=50000,num_threads=4,min_after_dequeue=10000)



#FEAT_TYPE = 'mfcc_fft512_hop160_vad_cmn'
FEAT_TYPE = FEAT_TYPE.split('_exshort')[0]
FEAT_TYPE = FEAT_TYPE.split('_fixed')[0]
# DATA_NAME = DATA_NAME.split('dev')[0]+'test'
records_test_list = []
for i in range(1,2):
    records_test_list.append(TFRECORDS_FOLDER+TEST_SET_NAME+'_'+FEAT_TYPE+'.'+str(i)+'.tfrecords')


#data for validation
vali_labels,vali_shapes,vali_feats = read_and_decode_emnet_mfcc(records_test_list)
# vali_labels_batch,vali_feats_batch,vali_shapes_batch = tf.train.batch(
#     [vali_labels, vali_feats, vali_shapes], batch_size=BATCHSIZE*2, dynamic_pad=True, allow_smaller_final_batch=True,
#     capacity=50,num_threads=1)



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
#     tst_list.extend([voxceleb_dir+'/voxceleb1_wav/'+key])

tst_enrolls_idx = []
tst_tests_idx = []
for index in range(len(tst_enrolls)):
    tst_enrolls_idx = np.append(tst_enrolls_idx, int(tst_dict[tst_enrolls[index]]))
    tst_tests_idx = np.append(tst_tests_idx, int(tst_dict[tst_tests[index]]))

print np.shape(tst_list)
print np.shape(tst_trials), np.shape(tst_enrolls), np.shape(tst_enrolls_idx)


### Initialize network related variables

with tf.device('/cpu:0'):

    softmax_num = SOFTMAX_NUM
    global_step = tf.Variable(0, trainable=False)
#     learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step,
#                                                50000, 0.98, staircase=True)

    opt = tf.train.RMSPropOptimizer(LEARNING_RATE)

    emnet_losses = []
    emnet_grads = []

#     feat_batch = tf.placeholder(tf.float32, [None,None,np.int(INPUT_DIM)],name="feat_batch")
#     label_batch = tf.placeholder(tf.int32, [None],name="label_batch")
#     shape_batch = tf.placeholder(tf.int32, [None,2],name="shape_batch")

    test_feat_batch = tf.placeholder(tf.float32, [None,None,np.int(INPUT_DIM)],name="test_feat_batch")
    test_label_batch = tf.placeholder(tf.int32, [None],name="test_label_batch")
    test_shape_batch = tf.placeholder(tf.int32, [None,2],name="test_shape_batch")



    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(NUMGPUS):
            with tf.device('/gpu:%d' % i):
                emnet = nn_model.nn(feats_batch, labels_batch,labels_batch, shapes_batch, softmax_num,True,INPUT_DIM,is_batchnorm)
#                 emnet = nn_model.nn(feat_batch,label_batch,label_batch,shape_batch, softmax_num,True,INPUT_DIM,is_batchnorm)
                tf.get_variable_scope().reuse_variables()
                grads = opt.compute_gradients(emnet.loss)
                emnet_losses.append(emnet.loss)
                emnet_grads.append(grads)

        with tf.device('/gpu:0'):
#             emnet_validation = nn_model.nn(vali_feats_batch,vali_labels_batch,vali_labels_batch,vali_shapes_batch, softmax_num,False,INPUT_DIM,is_batchnorm);
            emnet_validation = nn_model.nn(test_feat_batch,test_label_batch,test_label_batch,test_shape_batch, softmax_num,False,INPUT_DIM,is_batchnorm);
            tf.get_variable_scope().reuse_variables()

    loss = tf.reduce_mean(emnet_losses)
    grads = average_gradients(emnet_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver(max_to_keep=MAX_SAVEFILE_LIMIT)


#    summary_writer = tf.summary.FileWriter(SAVER_FOLDERNAME, sess.graph)
    #variable_summaries(loss)

#    tf.summary.scalar('loss', loss)
#    tf.summary.histogram('loss', loss)
#    acc_tf = tf.Variable(tf.zeros([1]),name='acc')
#    tf.summary.scalar('Accuracy_tst', tf.squeeze(acc_tf))

#    summary_op = tf.summary.merge_all()

    tf.initialize_all_variables().run()
    tf.train.start_queue_runners(sess=sess)

    #load entire test set
    test_list = np.loadtxt('data/voxceleb1_test/wav.scp',dtype='string',usecols=[1])
    test_feats =[]
    test_labels = []
    test_shapes = []
    for iter in range(len(test_list)):
        a,b,c = sess.run([vali_feats,vali_labels,vali_shapes])
        test_feats.extend([a])
        test_labels.extend([b])
        test_shapes.extend([c])

    ### Training neural network
#     resume=False
#     START=0
    if resume:
        saver.restore(sess,SAVER_FOLDERNAME+'/model'+str(RESUME_STARTPOINT)+'.ckpt-'+str(RESUME_STARTPOINT))

    for step in range(RESUME_STARTPOINT,MAX_ITER):
#         a,b,c = sess.run([feats_batch,labels_batch,shapes_batch])

#         if np.shape(a)[1]<INPUT_LENGTH:
#             new_a = a
#         else:
#             new_a = np.zeros([len(a),INPUT_LENGTH,np.int(INPUT_DIM)])
#             for iter in range(len(a)):
#                 a_start = 0
#                 a_end = INPUT_LENGTH
#                 if c[iter,0]>INPUT_LENGTH:
#                     a_start = np.random.randint(0,c[iter,0]-INPUT_LENGTH,1)[0]
#                     a_end = a_start +INPUT_LENGTH
#                     c[iter,0] = INPUT_LENGTH
#                 new_a[iter,:,:] = a[iter,a_start:a_end,:]

#         _, loss_v,mean_loss = sess.run([apply_gradient_op, emnet.loss,loss],feed_dict={
#        	    feat_batch: new_a,
# 	    label_batch: b,
# 	    shape_batch: c
#         })
        _, loss_v,mean_loss = sess.run([apply_gradient_op, emnet.loss,loss])


        if np.isnan(loss_v):
            print ('Model diverged with loss = NAN')
            quit()

        if step % EERTEST_INTERVAL ==0 and step>=RESUME_STARTPOINT:
            embeddings = []

            for iter in range(len(test_list)):
                eb = emnet_validation.eb.eval({test_feat_batch:[test_feats[iter]], test_label_batch:[test_labels[iter]], test_shape_batch:[test_shapes[iter]]})
                embeddings.extend([eb])
            embeddings = np.squeeze(embeddings)

            scores = np.zeros([len(tst_enrolls_idx)])
            for iter in range(len(tst_enrolls_idx)):
                scores[iter] =  embeddings[int(tst_enrolls_idx[iter]),:].dot( embeddings[int(tst_tests_idx[iter]),:].transpose())

            tst_trials = map(int,tst_trials)
            fpr,tpr,threshold = roc_curve(tst_trials,scores,pos_label=1)
            fnr = 1-tpr
            EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
            EER = fpr[np.argmin(np.absolute((fnr-fpr)))]
            print ('Step %d: loss %.3f, lr : %.5f, EER : %f' % (step,mean_loss, LEARNING_RATE,EER))


        if step % TESTSET_INTERVAL ==0 and step >=RESUME_STARTPOINT:
            prediction = np.empty((0,5))
            _label = np.int64([])
            a,b,c = sess.run([feats_batch,labels_batch,shapes_batch])
#             d,e,f = sess.run([vali_feats_batch,vali_labels_batch,vali_shapes_batch])

            prediction, _label= sess.run([emnet_validation.o1, emnet_validation.label ], feed_dict={
                test_feat_batch:a,
                test_label_batch:b,
                test_shape_batch:c
            })
            spklab_num_mat = np.eye(softmax_num)[_label]
            acc = accuracy(prediction, spklab_num_mat)
            print ('Step %d: loss %.3f, lr : %.5f, Accuracy : %f' % (step,mean_loss, LEARNING_RATE,acc))
#             acc_op = acc_tf.assign([acc])

        if step % LOSS_INTERVAL ==0:
            print ('Step %d: loss %.3f, lr : %.5f' % (step, mean_loss, LEARNING_RATE))
#             summary_str, _ = sess.run([summary_op, acc_op])
#             summary_writer.add_summary(summary_str,step)

        if step % SAVE_INTERVAL == 0 and step >=RESUME_STARTPOINT:
            saver.save(sess, SAVER_FOLDERNAME+'/model'+str(step)+'.ckpt',global_step=step)
