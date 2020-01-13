import numpy as np
import os,sys
sys.path.insert(0,'scripts/')
import kaldi_data as kd
from sklearn.cluster import AgglomerativeClustering

from tqdm import tqdm




import argparse
parser = argparse.ArgumentParser(description="To train speaker verification network using voxceleb1", add_help=True)
parser.add_argument("--max_spks", type=int, default=5,help="source data")
parser.add_argument("--data_folder", type=str, default='data/adi_test',help="target data")
parser.add_argument("--total_split", type=int, default=2,help="target data")
parser.add_argument("--embedding_folder", type=str, default = 'exp/embeddings',help="target data")
parser.add_argument("--model_name", type=str,default='spk2vec_test24_aug', help="target data")
# parser.add_argument("--segments_format", type=str, default='True', help="sgd,rms or adam")
parser.add_argument("--embedding_layer", type=str, default='fc2', help="sgd,rms or adam")

args = parser.parse_known_args()[0]
print args

args.embedding_layer = args.embedding_layer.split('/')[-1]

max_spks = args.max_spks
mat = []
seg_clusters = []

for split in range(1,args.total_split+1):
    temp = np.load(args.embedding_folder+'/'+args.data_folder.split('/')[-1]+'_'+args.model_name+'.'+str(split)+'.'+args.embedding_layer+'.npy')
    mat.extend(temp)

mat = np.array(mat)

#length normalization
for iter in range(len(mat)):
    mat[iter] = mat[iter]/np.linalg.norm(mat[iter])

#read segments file
segments = open(args.data_folder+'/segments').readlines()
uttid = []
for line in segments:
    uttid.append(line.split()[1])
uttid = np.array(uttid)

#clustering with AHC
for iter in range(len(np.unique(uttid))):
    _,inverse_idx = np.unique(uttid,return_inverse=True)
    idx = np.nonzero(inverse_idx==iter)[0]
    if len(idx)<3:
        clusters = np.zeros([len(idx)],dtype=int)            
    else:
        if len(idx)>=10:  
            spks = max_spks
        else:
            spks = len(idx)/2

        X = mat[idx,:]
        clustering = AgglomerativeClustering(n_clusters=spks,affinity='cosine',linkage='single').fit(X)
        clusters = clustering.labels_
    seg_clusters.extend(clusters)

# file write
fid = open(args.data_folder+'/seg2spk','w')
for iter in range(len(segments)):
    fid.write('%s %d\n'%(segments[iter].rstrip(),seg_clusters[iter]))
fid.close()
