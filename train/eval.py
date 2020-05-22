import os
import numpy as np
import torch
from torch import nn
import time
import datetime
import pickle
import tqdm
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors

from model import Model

class Solver(object):
    def __init__(self, data_path, model_path, batch_size, is_cuda=True):
        self.data_path = data_path
        self.batch_size = batch_size
        self.is_cuda = is_cuda
        self.model = Model()
        self.load_test_files()
        self.load(model_path)
        if self.is_cuda:
            self.model = self.model.cuda()

    def load_test_files(self):
        test_gt_fn = '../msd_sub/test_gt.npy'
        self.test_list = np.load('../msd_sub/test.npy')
        self.test_list.sort()
        track_tags = pd.read_csv('../msd_sub/track_tags.tsv', sep='\t', names=['id', 'tag', 'cleaned', 'type', 'number'])
        self.tags = list(set(track_tags.cleaned))
        self.tags.sort()
        self.glove = pickle.load(open('../msd_sub/glove_200d.pkl', 'rb'))
        self.glove_embs = np.zeros((100, 200))
        for i in range(100):
            self.glove_embs[i] = self.glove[self.tags[i]]
        self.glove_embs = torch.tensor(self.glove_embs.astype('float32'))
        if os.path.exists(test_gt_fn):
            self.test_gt = np.load(test_gt_fn)
        else:
            print('get test ground truth binaries')
            self.test_gt = np.zeros((len(self.test_list), len(self.tags)))
            i = 0
            for _song in tqdm.tqdm(self.test_list):
                annot = list(track_tags[track_tags.id==_song].cleaned)
                for _tag in annot:
                    self.test_gt[i][self.tags.index(_tag)] = 1
                i += 1
            np.save(open(test_gt_fn, 'wb'), self.test_gt)
            print('done!')

    def to_var(self, x):
        if self.is_cuda:
            x = x.cuda()
        return x

    def load(self, filename):
        S = torch.load(filename)
        SS = {}
        for k, v in S['model'].items():
            SS[k] = v
        self.model.load_state_dict(SS)
        print('model %s loaded' % os.path.basename(filename)[:-4])

    def evaluation(self):
        self.model.eval()
        # song to embedding
        song_embs = np.zeros((len(self.test_list), 256))
        song_per_batch = 4
        num_iter = len(self.test_list) // song_per_batch
        remainder = len(self.test_list) % song_per_batch
        for _iter in tqdm.tqdm(range(num_iter)):
            indice = np.arange(song_per_batch) + (_iter * song_per_batch)
            song_embs[indice] = self.song_to_emb(indice, song_per_batch)
        if remainder > 0:
            indice = np.arange(len(self.test_list) - remainder, len(self.test_list))
            song_embs[indice] = self.song_to_emb(indice, remainder)
        song_embs = torch.tensor(song_embs.astype('float32'))

        # tag to embedding
        tag_embs = self.tags_to_emb()

        # get similarity score
        sim_scores = self.get_similarity(tag_embs, song_embs)

        # get evaluation metrics
        k = 10
        p_ks = self.get_precision(sim_scores, k=k)
        roc_aucs = self.get_roc_auc(sim_scores)
        aps = self.get_ap(sim_scores)

        print('precision @%d: %.4f' % (k, np.mean(p_ks)))
        print('roc_auc: %.4f' % np.mean(roc_aucs))
        print('map: %.4f' % np.mean(aps))

        for i, tag in enumerate(self.tags):
            print('%s: %.4f, %.4f, %.4f' % (tag, p_ks[i], roc_aucs[i], aps[i]))

        return p_ks, roc_aucs, aps, sim_scores

    def song_to_emb(self, indice, song_per_batch):
        batch_size = 8
        length = 256
        batch = np.zeros((batch_size * song_per_batch, 128, length))
        for i, ix in enumerate(indice):
            song_id = self.test_list[ix]
            fn = os.path.join('/ssd2/dataset/msd/spec_clean', song_id[2], song_id[3], song_id[4], song_id+'.npy')
            spec = np.load(fn)
            hop = (spec.shape[1] - length) // batch_size
            for _b in range(batch_size):
                batch[i*batch_size + _b] = spec[:, _b*hop:_b*hop+length]
        batch = self.to_var(torch.tensor(batch.astype('float32')))
        emb = self.model.spec_to_embedding(batch)
        song_embs = torch.zeros(song_per_batch, 256)
        for i in range(song_per_batch):
            song_emb = emb[i*batch_size:(i+1)*batch_size].mean(axis=0).unsqueeze(0)
            song_embs[i] = song_emb.cpu().detach()
        return song_embs

    def tags_to_emb(self):
        word_emb = self.model.word_to_embedding(self.to_var(self.glove_embs)).cpu().detach()
        return word_emb

    def get_similarity(self, tag_embs, song_embs):
        sim_scores = np.zeros((len(tag_embs), len(song_embs)))
        for i in range(100):
            sim_scores[i] = np.array(nn.CosineSimilarity(dim=-1)(tag_embs[i], song_embs))
        return sim_scores

    def get_precision(self, sim_scores, k=10):
        p_ks = []
        for i in range(100):
            sorted_ix = np.argsort(sim_scores[i])[::-1][:k]
            gt = self.test_gt.T[i][sorted_ix]
            p_k = metrics.precision_score(gt, np.ones(k))
            p_ks.append(p_k)
        return p_ks

    def get_roc_auc(self, sim_scores):
        roc_aucs = metrics.roc_auc_score(self.test_gt, sim_scores.T, average=None)
        return roc_aucs

    def get_ap(self, sim_scores):
        ap = metrics.average_precision_score(self.test_gt, sim_scores.T, average=None)
        return ap

s = Solver('/ssd2/dataset/msd', 'model/model_266.pth', 64)
p_ks, roc_aucs, aps, sim_scores = s.evaluation()

#d = {}
#d['p_10'] = p_ks
#d['roc_auc'] = roc_aucs
#d['pr_auc'] = aps
#pickle.dump(d, open('metrics.pkl', 'wb'))
