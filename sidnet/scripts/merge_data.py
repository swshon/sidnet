import sys
sys.path.insert(0,'scripts/')
import kaldi_data as kd

import argparse
parser = argparse.ArgumentParser(description="Merge two data folder to target folder", add_help=True)
parser.add_argument("--target", type=str, default="data/voxceleb1",help="source data")
parser.add_argument("--source1", type=str, default="data/voxceleb1_dev", help="target data")
parser.add_argument("--source2", type=str, default="data/voxceleb1_test", help="target data")
parser.add_argument("--utt2spk", action='store_true', help="for utt2spk file")
parser.add_argument("--utt2lang", action='store_true', help="for utt2lang file")
args = parser.parse_known_args()[0]

wavlist = []
utt_label = []
spk_label = []
lang_label = []

for name in [args.source1, args.source2]:
    print name
    if args.utt2spk:
        if args.utt2lang:
            wav,utt,spk,lang = kd.read_data_list(name, utt2spk=True, utt2lang=True)
            lang_label.extend(lang)
        else:
            wav,utt,spk = kd.read_data_list(name, utt2spk=True)
        spk_label.extend(spk)
    elif args.utt2lang:
        wav,utt,lang = kd.read_data_list(name, utt2lang=True)
        lang_label.extend(lang)

    wavlist.extend(wav)
    utt_label.extend(utt)



# if args.utt2lang:
#     wavlist,utt_label,lang_label = kd.read_data_list(args.source1,utt2lang=True)
#
# if args.utt2spk:
#     kd.split_data(SOURCE_FOLDER,wavlist,utt_label,spk_label=spk_label,total_split=TOTAL_SPLIT)
#
# if args.utt2lang:
#     kd.split_data(SOURCE_FOLDER,wavlist,utt_label,lang_label=lang_label,total_split=TOTAL_SPLIT)




# kd.write_data(TARGET_FOLDER,wavlist,utt_label,spk_label)
if args.utt2spk:
    if args.utt2lang:
        kd.write_data(args.target,wavlist,utt_label,spk_label=spk_label,lang_label=lang_label)
    else:
        kd.write_data(args.target,wavlist,utt_label,spk_label=spk_label)
elif args.utt2lang:
    kd.write_data(args.target,wavlist,utt_label,lang_label=lang_label)
