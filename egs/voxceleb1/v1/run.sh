#!/bin/bash
export LC_ALL=C
stage=1

if [ $stage -eq 0 ]; then
  ln -s ../../../sidnet/scripts scripts
  ln -s ../../../sidnet/models models
fi

if [ $stage -eq 1 ]; then
  ./local/make_voxceleb1_sv.pl /data/sls/SID/Corpora/voxceleb1/ data
  #python ./scripts/shuffle_data.py --source data/voxceleb1_dev --target data/voxceleb1_dev_shuffle --utt2spk
  #python ./scripts/split_data.py --source data/voxceleb1_dev_shuffle --split 100 --utt2spk
  python ./scripts/split_data.py --source data/voxceleb1_dev --split 100 --utt2spk
  python ./scripts/split_data.py --source data/voxceleb1_test --split 1 --utt2spk

  #Pick first utterance per speakers for validation purpose
  python ./local/pick_1utt_per_1spk.py --source data/voxceleb1_dev --target data/voxceleb1_dev_1utt

fi

if [ $stage -eq 2 ]; then
# Extract feature for NN input and save in tfrecords format for Tensorflow
mkdir -p data/tfrecords
mkdir -p log/feats

TOTAL_SPLIT=100
for data in voxceleb1_dev; do
  for (( split=1; split<=$TOTAL_SPLIT; split++ )); do
    echo $split
    srun -J logmel_${data} -p 630 --exclude=sls-630-5-1 --cpus-per-task=2 --mem=24GB \
      --output=log/feats/logmel_${data}_${split}.out \
      python scripts/extract_feat_tfrecords.py \
      --feat_type logmel \
      --feat_dim 40 \
      --nfft 512 \
      --win_len 400 \
      --hop 160 \
      --vad True \
      --cmvn mv \
      --data_folder data/$data \
      --total_split $TOTAL_SPLIT \
      --current_split $split \
      --save_folder data/tfrecords \
      --utt2label utt2spk \
      --fixed_len 798 &
  done
done
fi
