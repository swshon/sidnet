#!/bin/bash
export LC_ALL=C
stage=1
feat_type=logmel

if [ $stage -eq 0 ]; then
  # Prepare scripts
  ln -s ../../../sidnet/scripts scripts
  ln -s ../../../sidnet/models models
#  ln -s ../../../../../../../temp/voxceleb1_dev_1utt_logmel_win400_hop160_vad_fixed598.1.tfrecords voxceleb1_dev_1utt_logmel_win400_hop160_vad_fixed598.1.tfrecords
#  ln -s ../../../../../../../temp/voxceleb1_test_logmel_win400_hop160_vad.1.tfrecords voxceleb1_test_logmel_win400_hop160_vad.1.tfrecords
fi

if [ $stage -eq 1 ]; then
  # Preprare voxceleb1 data
  ./local/make_voxceleb1_sv.pl /data/sls/SID/Corpora/voxceleb1/ data

  # if you need shuffle (you don't need to shuffle)
  #python ./scripts/shuffle_data.py --source data/voxceleb1_dev --target data/voxceleb1_dev_shuffle --utt2spk
  #python ./scripts/split_data.py --source data/voxceleb1_dev_shuffle --split 100 --utt2spk

  # Split data into multiple for parallel jobs
  python ./scripts/split_data.py --source data/voxceleb1_dev --split 100 --utt2spk
  python ./scripts/split_data.py --source data/voxceleb1_test --split 1 --utt2spk

  # Pick first utterance per speakers for validation purpose
  python ./local/pick_1utt_per_1spk.py --source data/voxceleb1_dev --target data/voxceleb1_dev_1utt
  python ./scripts/split_data.py --source data/voxceleb1_dev_1utt --split 1 --utt2spk

fi

if [ $stage -eq 2 ]; then
  # Extract for NN input and save in tfrecords format for Tensorflow
  mkdir -p data/tfrecords
  mkdir -p log/feats
  TOTAL_SPLIT=100

  for data in voxceleb1_dev ; do
    for (( split=1; split<=$TOTAL_SPLIT; split++ )); do
      echo $split
      srun -J ${feat_type}_${data} -p 630 --exclude=sls-630-5-1 --cpus-per-task=2 --mem=24GB \
        --output=log/feats/${feat_type}_${data}_${split}.out \
        python scripts/extract_feat_tfrecords.py \
        --feat_type $feat_type \
        --feat_dim 40 \
        --nfft 512 \
        --win_len 400 \
        --hop 160 \
        --vad True \
        --cmvn False \
        --data_folder data/$data \
        --total_split $TOTAL_SPLIT \
        --current_split $split \
        --save_folder data/tfrecords \
        --utt2label utt2spk \
        --fixed_len 598 &
    done
  done
fi


if [ $stage -eq 3 ]; then
  mkdir -p data/tfrecords
  mkdir -p log/feats
  data=voxceleb1_test
  srun -J ${feat_type}_${data} -p 630 --exclude=sls-630-5-1 --cpus-per-task=2 --mem=24GB \
  --output=log/feats/${feat_type}_${data}_1.out \
  python scripts/extract_feat_tfrecords.py \
  --feat_type $feat_type \
  --feat_dim 40 \
  --nfft 512 \
  --win_len 400 \
  --hop 160 \
  --vad True \
  --cmvn False \
  --data_folder data/$data \
  --total_split 1 \
  --current_split 1 \
  --save_folder data/tfrecords \
  --utt2label utt2spk &
fi


if [ $stage -eq 4 ]; then
  # Extract for NN input and save in tfrecords format for Tensorflow
  mkdir -p data/tfrecords
  mkdir -p log/feats
  TOTAL_SPLIT=1

  for data in voxceleb1_dev_1utt ; do
    for (( split=1; split<=$TOTAL_SPLIT; split++ )); do
      echo $split
      srun -J ${feat_type}_${data} -p 630 --exclude=sls-630-5-1 --cpus-per-task=2 --mem=24GB \
        --output=log/feats/${feat_type}_${data}_${split}.out \
        python scripts/extract_feat_tfrecords.py \
        --feat_type $feat_type \
        --feat_dim 40 \
        --nfft 512 \
        --win_len 400 \
        --hop 160 \
        --vad True \
        --cmvn False \
        --data_folder data/$data \
        --total_split $TOTAL_SPLIT \
        --current_split $split \
        --save_folder data/tfrecords \
        --utt2label utt2spk \
        --fixed_len 598 &
    done
  done
fi

if [ $stage -eq 5 ]; then
  mkdir -p saver
  echo "Training resnet network using Logmel with softmax, Momentum, test +vad + variable 200~400 len"
  cp models/spk2vec_resnet_ver17_ams_att.py models/spk2vec_resnet_ver17_ams_att_momentum_vad_randsubsample.py

  model=spk2vec_resnet_ver17_ams_att_momentum_vad_randsubsample
  feat_type=logmel
  feat_spec=win400_hop160_vad_fixed598
  srun -J v17ams_${feat_type} -p sm,2080,1080 --gres=gpu:1 --exclude=$ex_machines --cpus-per-task=2 --mem=16GB \
  --output=./log/${model}_${feat_type}_${feat_spec} \
  python ./scripts/train_sidnet.py \
  --model_name $model \
  --feat_type ${feat_type}_${feat_spec} \
  --max_iteration 6000000 \
  --optimizer momentum \
  --print_loss_interval 100 \
  --learning_rate 0.005 \
  --mini_batch 4 \
  --cnn1d false \
  --main_scope resnet_v2_softmax \
  --fixed_input_frame 598 \
  --input_dim 40 \
  --subsample_min 200 \
  --subsample_max 400 \
  --train_data voxceleb1_dev_1utt \
  --train_total_split 1 \
  --test_data voxceleb1_test \
  --print_eer_interval 4000 \
  --print_loss_interval 1  \
  --print_train_acc_interval 10 \
  --momentum 0.9 \
  --embedding_scope fc2 &
fi


if [ $stage -eq 6 ]; then
  mkdir -p saver
  echo "Training resnet network using Logmel with softmax, Momentum, test +vad + variable 200~400 len"
  cp models/spk2vec_resnet_ver17_ams_att.py models/spk2vec_resnet_ver17_ams_att_momentum_vad_randsubsample.py

  model=spk2vec_resnet_ver17_ams_att_momentum_vad_randsubsample
  feat_type=logmel
  feat_spec=win400_hop160_vad_fixed598
  srun -J v17ams_${feat_type} -p sm,2080,1080 --gres=gpu:1 --exclude=$ex_machines --cpus-per-task=2 --mem=16GB \
  --output=./log/${model}_${feat_type}_${feat_spec} \
  python ./scripts/train_sidnet.py \
  --model_name $model \
  --feat_type ${feat_type}_${feat_spec} \
  --max_iteration 6000000 \
  --optimizer momentum \
  --print_loss_interval 100 \
  --learning_rate 0.005 \
  --mini_batch 4 \
  --cnn1d false \
  --main_scope resnet_v2_softmax \
  --fixed_input_frame 598 \
  --input_dim 40 \
  --subsample_min 200 \
  --subsample_max 400 \
  --train_data voxceleb1_dev \
  --train_total_split 100 \
  --test_data voxceleb1_test \
  --print_eer_interval 4000 \
  --print_loss_interval 1  \
  --print_train_acc_interval 10 \
  --momentum 0.9 \
  --embedding_scope fc2 &
fi
