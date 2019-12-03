import sys
sys.path.insert(0,'scripts/')
import kaldi_data as kd

import argparse
parser = argparse.ArgumentParser(description="Split data", add_help=True)
parser.add_argument("--source", type=str, default="data/adi_test",help="source data")
parser.add_argument("--split", type=str, default="2", help="target data")
# parser.add_argument("--utt2spk", action='store_true', help="for utt2spk file")
# parser.add_argument("--utt2lang", action='store_true', help="for utt2lang file")
args = parser.parse_known_args()[0]

SOURCE_FOLDER = args.source
TOTAL_SPLIT = int(args.split)

# BASE_FOLDER = sys.argv[1]
# TOTAL_SPLIT = int(sys.argv[2])

    
segments = open(SOURCE_FOLDER+'/segments').readlines()
kd.split_segments(SOURCE_FOLDER,segments,TOTAL_SPLIT)



