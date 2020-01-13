import os,sys
import numpy as np
sys.path.insert(0, './scripts')
sys.path.insert(0, './models')
import nn_utils as utils
from tqdm import tqdm
import kaldi_data as kd

class validation_set():
    def __init__(self, args):
        validation_data_name = args.data_root +'/'+args.test_data
        trials_filename = args.data_root +'/'+args.test_trials

        # Load validation trials
        tst_trials=[]
        tst_enrolls = []
        tst_tests = []

        for line in open(trials_filename,'r').readlines():
            row = line.split()
            tst_enrolls.append(row[0])
            tst_tests.append(row[1])
            tst_trials.append(np.where(row[2]=='target',1,0))
        tst_trials = np.array(tst_trials)

        wavlist,utt_label,spk_label = kd.read_data_list(validation_data_name, utt2spk=True, utt2lang=False)
        tst_dict=dict()
        tst_list=[]
        for value,key in enumerate(utt_label):
            tst_dict[key]=value

        tst_enrolls_idx = np.array([],dtype = int)
        tst_tests_idx = np.array([],dtype = int)
        for index in range(len(tst_enrolls)):
            tst_enrolls_idx = np.append(tst_enrolls_idx, int(tst_dict[tst_enrolls[index]]))
            tst_tests_idx = np.append(tst_tests_idx, int(tst_dict[tst_tests[index]]))

        self.wavlist = wavlist
        self.tst_enrolls_idx = tst_enrolls_idx
        self.tst_tests_idx = tst_tests_idx
        self.data=validation_data_name
        self.tst_trials = tst_trials
        self.args = args

    def load_feature(self,sess, vali_labels, vali_shapes, vali_feats ): # get mean vector based on the YouTube ID
        test_feats =[]
        test_labels = []
        test_shapes = []
        for iter in tqdm(range(len(self.wavlist))):
        #     for iter in range(10):
            a,b,c = sess.run([vali_feats,vali_labels,vali_shapes])
            test_feats.extend([a])
            test_labels.extend([b])
            test_shapes.extend([c])
        self.feats = test_feats
        self.labels = test_labels
        self.shapes = test_shapes

    def get_scores(self, emnet_validation,feat_batch,label_batch):
        valid_list= np.unique(np.append(self.tst_enrolls_idx,self.tst_tests_idx))

        #Extract embeddings
        embeddings = []
        valid_list_inverse= dict()
        # for iter in tqdm(range(len(self.feats))):
        for count, idx in enumerate(tqdm(valid_list)):
            valid_list_inverse[idx] = count
            eb = emnet_validation.end_points[self.args.main_scope+'/'+self.args.embedding_scope].eval({feat_batch:[self.feats[idx]], label_batch:[self.labels[idx]]})
            embeddings.extend([eb])
        embeddings = np.squeeze(embeddings)

        #get cosine similarity scores
        scores = np.zeros([len(self.tst_enrolls_idx)])
        for iter in range(len(self.tst_enrolls_idx)):
            scores[iter] =  embeddings[valid_list_inverse[int(self.tst_enrolls_idx[iter])],:].dot( embeddings[valid_list_inverse[int(self.tst_tests_idx[iter])],:].transpose())

        #length normalization on embeddings
        norms = np.linalg.norm(embeddings,axis=1)
        for iter, norm in enumerate(norms):
            embeddings[iter,:] = embeddings[iter,:]/norm

        #get cosine silimiarity scores from lengh-norm embeddings
        norm_scores = np.zeros([len(self.tst_enrolls_idx)])
        for iter in range(len(self.tst_enrolls_idx)):
            # norm_scores[iter] =  embeddings[int(self.tst_enrolls_idx[iter]),:].dot( embeddings[int(self.tst_tests_idx[iter]),:].transpose())
            norm_scores[iter] =  embeddings[valid_list_inverse[int(self.tst_enrolls_idx[iter])],:].dot( embeddings[valid_list_inverse[int(self.tst_tests_idx[iter])],:].transpose())

        self.scores = scores
        self.norm_scores = norm_scores
