import numpy as np
import os,sys
sys.path.insert(0,'scripts/')
import shutil
import kaldi_data as kd

import argparse
parser = argparse.ArgumentParser(description="Shuttling data", add_help=True)
parser.add_argument("--source", type=str, default="data/test_segments/utt2lang_sorted",help="source data")
parser.add_argument("--target", type=str, default="result_test_sorted.csv", help="target data")
parser.add_argument("--utt2spk", action='store_true', help="for utt2spk file")
parser.add_argument("--utt2lang", action='store_true', help="for utt2lang file")
args = parser.parse_known_args()[0]

SOURCE_FOLDER = args.source
TARGET_FOLDER = args.target

if not os.path.exists(TARGET_FOLDER):
    os.mkdir(TARGET_FOLDER)
    
wavlist,utt_label,spk_label = kd.read_data_list(SOURCE_FOLDER, utt2spk=args.utt2spk, utt2lang=args.utt2lang)

idx = range(len(wavlist))
np.random.shuffle(idx)

wavlist = wavlist[idx]
utt_label = utt_label[idx]
spk_label = spk_label[idx]
kd.write_data(TARGET_FOLDER,wavlist,utt_label,spk_label)

