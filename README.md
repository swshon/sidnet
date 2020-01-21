# Speaker ID NETwork (SIDNET) Tool
This repository provides SIDNET Tool for the speaker verification task. This toolkit does not rely on Kaldi although it has almost similar data prepration format to Kaldi. This toolkit was originally written to analyze the internal representation of speaker recognition network [1].

# Tutorial ( egs/voxceleb1)
First, you need to download dataset and clone this repository. Then run

    cd egs/voxceleb1/v1
    ./run.sh
Note that this tutorial does not use any speech or audio data other than voxceleb1 data for benchmark purpose. Hence, there is no noise augmentation on the training data.

A brief features of the training scheme is below

    Used Log-mel filterbanks (40 dimension) as input of the NN (512 fft, 400 samples per window, 160 samples stride)
    Cepstral mean normalization on log-mel filterbanks
    Voice Activity Detection with simple power threshold
    All training speech was subsampled between 2~4 seconds randomly
    Optimized with SGD (start from 0.005 learning rate)+Momentum (with factor of 0.9)
    Learning rate decay by 1/10 on every 5 epoch
    Mini-batch size = 16
    speaker embedding dimension = 512
    Training took roughly 12 hours on Titan X Pascal (training set has roughly 200h)


# Performance evaluation on Voxceleb1 test benchmark test using voxceleb1 training set (EER)
Scoring was done using Cosine similarity. Note these result only use voxceleb1 development dataset (total 1211 speakers) for training. There's no training data augmentation using noise or Room Impulse Response (RIR).

    5 layer CNN + Softmax: 7.06%
    5 layer CNN + Additive Margin Softmax (AMS) : 6.16%
    Resnet-50 + Softmax : 7.33%
    Resnet-50 + AMS :6.10%
    REsnet-50 + AMS + Self Attention Pooling (SAP) : 5.73%    

For comparison,

    Nagrani et al. (VGG-M): 7.82%
    Hajibabaei et al. (Temporal average pooling, Cosine Similarity , Resnet20, AMS, Augmentation): 4.30%
    Okabe et al. (x-vector, PLDA, Softmax, SAP, Augmentation) : 3.85%
    Chung et al. (Thin Resnet-34, SAP, Softmax, Augmentation): 5.71%

# Performance evaluation on Voxceleb1 test benchmark test using voxceleb1+2 training set (EER) (not yet updated this tutorial on egs folder)
Scoring was done using Cosine similarity. Note these result use voxceleb1 and voxceleb2 development dataset (total 7205 speakers) for training. There's no training data augmentation using noise or Room Impulse Response (RIR).

    REsnet-50 + AMS + Self Attention Pooling (SAP) : 2.78%

For comparison,

    Xie et al. (Thin Resnet-34, GhostVLAD, Softmax): 3.22%
    Xie et al. (Thin Resnet-34, GhostVLAD, AMS): 3.23%

# Requirements (for example training code and baseline code)
    Python 2.7
    tensorflow (python library, tested on 1.14)
    librosa (python library, tested on 0.6.0)

# Reference
    
[1] S. Shon, H. Tang, and J. Glass, "Frame-Level Speaker Embeddings for Text-Independent Speaker Recognition and Analysis of End-to-End Model," Proc. SLT, pp. 1007-1013, 2018

