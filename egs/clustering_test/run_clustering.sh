#!/bin/bash

export LC_ALL=C

stage=1

TOTAL_SPLIT=2
#Split data folder into multiple chunk
if [ $stage -eq 1 ]; then
python ./scripts/split_data_segments.py --source data/adi_test --split $TOTAL_SPLIT

fi


#Extract speaker embedding using Tensorflow
if [ $stage -le 2 ]; then
mkdir -p exp/embeddings
for (( split=1; split<=$TOTAL_SPLIT; split++ )); do
  python ./scripts/extract_embedding_old.py \
      --feat_type mfcc \
      --feat_dim 40 \
      --nfft 512 \
      --win_len 400 \
      --hop 160 \
      --vad True \
      --cmvn m \
      --data_folder data/adi_test \
      --total_split $TOTAL_SPLIT \
      --current_split $split \
      --save_folder exp/embeddings \
      --model_name spk2vec_test24_aug \
      --softmax_num 1211 \
      --resume_startpoint 6992000 \
      --segments_format True \
      --embeddings_layer softmax/fc2

done

fi

#Clustering and write seg2spk label file in data folder. The last column in seg2spk file represent speaker ID.
if [ $stage -le 3 ]; then

python ./scripts/spk_clustering.py \
      --max_spks 5 \
      --data_folder data/adi_test \
      --total_split $TOTAL_SPLIT \
      --embedding_folder exp/embeddings \
      --model_name spk2vec_test24_aug \
      --embedding_layer softmax/fc2

fi
