#!/data/sls/u/swshon/tools/pytf/bin/python

from pandas import read_table, DataFrame, concat
from glob import glob
from tqdm import tqdm

import os,sys
import tensorflow as tf
import numpy as np
import librosa

sys.path.insert(0, './scripts')
sys.path.insert(0, './models')

import feature_tools as ft

voxceleb_dir = '/data/sls/scratch/swshon/dataset/voxceleb1'
import kaldi_data as kd


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

import argparse
parser = argparse.ArgumentParser(description="extract feature and save in tfrecords format", add_help=True)
parser.add_argument("--feat_type", type=str,default='mfcc', help="mfcc/logmel")
parser.add_argument("--feat_dim", type=str,default='40', help="target data")
parser.add_argument("--nfft", type=str,default='512', help="target data")
parser.add_argument("--win_len", type=str,default='400', help="target data")
parser.add_argument("--hop", type=str, default='160',help="target data")
parser.add_argument("--vad", type=str, default='True',help="target data")
parser.add_argument("--cmvn", type=str, default='m',help="target data")
parser.add_argument("--exclude_short", type=str, default="0", help="target data")
parser.add_argument("--data_folder", type=str, default='data/adi_test',help="target data")
parser.add_argument("--total_split", type=str, default='1',help="target data")
parser.add_argument("--current_split", type=str, default = '1',help="target data")
parser.add_argument("--save_folder", type=str, default = 'exp/embeddings',help="target data")
# parser.add_argument("--utt2label", type=str, default='test',help="target data")
# parser.add_argument("--fixed_len", type=str, default="0", help="target data")
parser.add_argument("--model_name", type=str,default='spk2vec_test24_aug', help="target data")
parser.add_argument("--softmax_num", type=int, default=1211, help="target data")
parser.add_argument("--resume_startpoint", type=int, default=6992000, help="0 if you don't have to resume")
parser.add_argument("--segments_format", type=str, default='True', help="sgd,rms or adam")
parser.add_argument("--embedding_layer", type=str, default='softmax/fc2', help="sgd,rms or adam")

args = parser.parse_known_args()[0]
args.segments_format = str2bool(args.segments_format)

# Feature extraction configuration
FEAT_TYPE = args.feat_type
N_FFT = int(args.nfft)
HOP = int(args.hop)
VAD = args.vad
CMVN = args.cmvn
EXCLUDE_SHORT = int(args.exclude_short)
DATA_FOLDER = args.data_folder
TOTAL_SPLIT = args.total_split
CURRRENT_SPLIT = args.current_split
SAVE_FOLDER = args.save_folder
FEAT_DIM = int(args.feat_dim)
WIN_LENGTH = int(args.win_len)
# FIXED_LEN = int(args.fixed_len) #298
SOFTMAX_NUM = args.softmax_num
RESUME_STARTPOINT = args.resume_startpoint
NN_MODEL = args.model_name
EMBEDDING_LAYER = args.embedding_layer

if VAD =='False':
    VAD = False
if CMVN == 'False':
    CMVN = False
is_batchnorm = True

if not args.segments_format:
    if int(TOTAL_SPLIT)==1:
        wavlist,utt_label,spk_label = kd.read_data_list(DATA_FOLDER, utt2spk=True)
    else:
        wavlist,utt_label,spk_label = kd.read_data_list(DATA_FOLDER+'/split'+TOTAL_SPLIT+'/'+CURRRENT_SPLIT, utt2spk=True)
    feat, _, utt_shape, tffilename = ft.feat_extract(wavlist,FEAT_TYPE,N_FFT,HOP,VAD,CMVN,EXCLUDE_SHORT)
else:
    if int(TOTAL_SPLIT)==1:
        wavlist,utt_label,seg_wavlist,seg_segid,seg_uttid,seg_windows = kd.read_data_list(DATA_FOLDER, utt2spk=False,segments=True)
    else:
        wavlist,utt_label,seg_wavlist,seg_segid,seg_uttid,seg_windows = kd.read_data_list(DATA_FOLDER+'/split'+TOTAL_SPLIT+'/'+CURRRENT_SPLIT, utt2spk=False,segments=True)
    feat, _, utt_shape, tffilename = ft.feat_extract(seg_wavlist,FEAT_TYPE,N_FFT,HOP,VAD,CMVN,EXCLUDE_SHORT,seg_windows=seg_windows)


SAVER_FOLDERNAME = 'saver/'+NN_MODEL+'_'+tffilename
nn_model = __import__(NN_MODEL)

x = tf.placeholder(tf.float32, [None,None,FEAT_DIM])
y = tf.placeholder(tf.int32, [None])
s = tf.placeholder(tf.int32, [None,2])


emnet_validation = nn_model.nn(x,y,y,s, SOFTMAX_NUM,False,FEAT_DIM,is_batchnorm);
tf.get_variable_scope().reuse_variables()


sess = tf.InteractiveSession()

saver = tf.train.Saver()
tf.initialize_all_variables().run()
tf.local_variables_initializer().run()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
saver.restore(sess,SAVER_FOLDERNAME+'/model'+str(RESUME_STARTPOINT)+'.ckpt-'+str(RESUME_STARTPOINT))

embeddings =[]
for iter in tqdm(range(len(feat))):
    eb = emnet_validation.end_points[EMBEDDING_LAYER].eval({x:[feat[iter]], s:[utt_shape[iter]]})
    embeddings.extend([eb])
embeddings = np.squeeze(embeddings)
embedding_filename = SAVE_FOLDER+'/'+DATA_FOLDER.split('/')[-1]+'_'+ NN_MODEL +'.'+CURRRENT_SPLIT+'.'+EMBEDDING_LAYER.split('/')[-1]+'.npy'
np.save( embedding_filename ,embeddings)