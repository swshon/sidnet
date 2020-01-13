import sys
sys.path.insert(0,'scripts/')
import kaldi_data as kd

import argparse
parser = argparse.ArgumentParser(description="Split data", add_help=True)
parser.add_argument("--source", type=str, default="data/test_segments/utt2lang_sorted",help="source data")
parser.add_argument("--split", type=str, default="result_test_sorted.csv", help="target data")
parser.add_argument("--utt2spk", action='store_true', help="for utt2spk file")
parser.add_argument("--utt2lang", action='store_true', help="for utt2lang file")
args = parser.parse_known_args()[0]

SOURCE_FOLDER = args.source
TOTAL_SPLIT = int(args.split)

if args.utt2spk:
    wavlist,utt_label,spk_label = kd.read_data_list(SOURCE_FOLDER, utt2spk=True)
if args.utt2lang:
    wavlist,utt_label,lang_label = kd.read_data_list(SOURCE_FOLDER,utt2lang=True)

if args.utt2spk:
    kd.split_data(SOURCE_FOLDER,wavlist,utt_label,spk_label=spk_label,total_split=TOTAL_SPLIT)

if args.utt2lang:
    kd.split_data(SOURCE_FOLDER,wavlist,utt_label,lang_label=lang_label,total_split=TOTAL_SPLIT)

