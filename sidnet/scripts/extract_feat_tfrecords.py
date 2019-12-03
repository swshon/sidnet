import os,sys
import tensorflow as tf
import numpy as np
import feature_tools as ft


def write_tfrecords(feat, utt_label, utt_shape, tfrecords_name):
    writer = tf.io.TFRecordWriter(tfrecords_name)
    trIdx = range(np.shape(utt_label)[0])

    # iterate over each example
    for count,idx in enumerate(trIdx):
        feats = feat[idx].reshape(feat[idx].size)
        label = utt_label[idx]
        shape = utt_shape[idx]

        # construct the Example proto boject
        example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(
              # Features contains a map of string to Feature proto objects
              feature={
                # A Feature contains one of either a int64_list,
                # float_list, or bytes_list
                'labels': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label])),
                'shapes': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=shape)),
                'features': tf.train.Feature(
                    float_list=tf.train.FloatList(value=feats.astype("float32"))),
        }))
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)

    writer.close()
    print tfrecords_name+": total "+str(len(feat))+" feature and "+str(np.shape(utt_label))+" label saved"




import argparse
parser = argparse.ArgumentParser(description="extract feature and save in tfrecords format", add_help=True)
parser.add_argument("--feat_type", type=str, help="mfcc/logmel")
parser.add_argument("--feat_dim", type=str, help="target data")
parser.add_argument("--nfft", type=str, help="target data")
parser.add_argument("--win_len", type=str, help="target data")
parser.add_argument("--hop", type=str, help="target data")
parser.add_argument("--vad", type=str, help="target data")
parser.add_argument("--cmvn", type=str, help="target data")
parser.add_argument("--exclude_short", type=str, default="0", help="target data")
parser.add_argument("--data_folder", type=str, help="target data")
parser.add_argument("--total_split", type=str, help="target data")
parser.add_argument("--current_split", type=str, help="target data")
parser.add_argument("--save_folder", type=str, help="target data")
parser.add_argument("--utt2label", type=str, help="target data")
parser.add_argument("--fixed_len", type=str, default="0", help="target data")

args = parser.parse_known_args()[0]


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
DIM = int(args.feat_dim)
WIN_LENGTH = int(args.win_len)
FIXED_LEN = int(args.fixed_len) #298

if VAD =='False':
    VAD = False
if CMVN == 'False':
    CMVN = False

lines = open(DATA_FOLDER+'/'+args.utt2label).readlines()

label2idx = {}
labels = [x.rstrip().split()[-1] for x in lines]
unique_labels = np.unique(labels)
for idx in range(len(unique_labels)):
    label2idx[unique_labels[idx]]=int(idx)

lines=open(DATA_FOLDER+'/split'+TOTAL_SPLIT+'/'+CURRRENT_SPLIT+'/'+args.utt2label).readlines()
utt_label = []
for line in lines:
    label=line.rstrip().split()[1]
    utt_label.append(label2idx[label])

wav_list = []
lines=open(DATA_FOLDER+'/split'+TOTAL_SPLIT+'/'+CURRRENT_SPLIT+'/wav.scp').readlines()
for line in lines:
    cols = line.rstrip().split()
    wav_list.append(cols[1])

feat, _, utt_shape, tffilename = ft.feat_extract(wav_list,FEAT_TYPE,N_FFT,HOP,VAD,CMVN,EXCLUDE_SHORT,False,DIM,WIN_LENGTH)


if FIXED_LEN>0:
    for iter in range(len(feat)):
        if feat[iter].shape[0]<=FIXED_LEN:
            while feat[iter].shape[0]<=FIXED_LEN:
                feat[iter] = np.append(feat[iter],feat[iter],0)
            feat[iter] = feat[iter][0:FIXED_LEN,:]
        else:
            rstart = np.random.randint(0,feat[iter].shape[0]-FIXED_LEN,1)[0]
            rend = rstart + FIXED_LEN
            feat[iter] = feat[iter][rstart:rend,:]
        utt_shape[iter] = np.array(feat[iter].shape)

    TFRECORDS_NAME = SAVE_FOLDER+'/'+DATA_FOLDER.split('/')[-1] + '_' + tffilename + '_fixed'+str(FIXED_LEN) + '.' + CURRRENT_SPLIT + '.tfrecords'
else:
    TFRECORDS_NAME = SAVE_FOLDER+'/'+DATA_FOLDER.split('/')[-1] + '_' + tffilename + '.' + CURRRENT_SPLIT + '.tfrecords'

write_tfrecords(feat,utt_label,utt_shape,TFRECORDS_NAME)
