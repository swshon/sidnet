import numpy as np
import os,sys
sys.path.insert(0,'scripts/')
import kaldi_data as kd
import argparse
parser = argparse.ArgumentParser(description="Shuttling data", add_help=True)
parser.add_argument("--source", type=str, default="data/voxceleb1_dev",help="source data")
parser.add_argument("--target", type=str, default="data/voxceleb1_dev_1utt", help="target data")
args = parser.parse_known_args()[0]

if not os.path.exists(args.target):
    os.mkdir(args.target)

wavlist,utt_label,spk_label = kd.read_data_list(args.source, utt2spk=True, utt2lang=False)
_, idx = np.unique(spk_label,return_index=True)

wavlist = wavlist[idx]
utt_label = utt_label[idx]
spk_label = spk_label[idx]
kd.write_data(args.target,wavlist,utt_label,spk_label)
