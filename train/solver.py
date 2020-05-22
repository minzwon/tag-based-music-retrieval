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

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, anchor, positive, negative, size_average=True):
        cosine_positive = nn.CosineSimilarity(dim=-1)(anchor, positive)
        cosine_negative = nn.CosineSimilarity(dim=-1)(anchor, negative)
        losses = self.relu(self.margin - cosine_positive + cosine_negative)
        return losses.mean()


class Solver(object):
    def __init__(self, data_loader, config):
        self.batch_size = config.batch_size
        self.data_path = config.data_path
        self.data_loader = data_loader
        self.n_epochs = config.n_epochs
        self.lr = config.lr
        self.margin = config.margin
        self.log_step = config.log_step
        self.save_step = config.save_step
        self.model_save_path = config.model_save_path
        self.is_cuda = config.is_cuda
        self.model = Model()
        self.load_valid_files()
        if len(config.model_load_path) > 1:
            self.load(config.model_load_path)
        if self.is_cuda:
            self.model = self.model.cuda()
       # self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 0.1, momentum=0.9, weight_decay=1e-6)

    def load_valid_files(self):
        valid_gt_fn = '../msd_sub/valid_gt.npy'
        self.valid_list = np.load('../msd_sub/valid.npy')
        self.valid_list.sort()
        track_tags = pd.read_csv('../msd_sub/track_tags.tsv', sep='\t', names=['id', 'tag', 'cleaned', 'type', 'number'])
        self.tags = list(set(track_tags.cleaned))
        self.tags.sort()
        self.glove = pickle.load(open('../msd_sub/glove_200d.pkl', 'rb'))
        self.glove_embs = np.zeros((100, 200))
        for i in range(100):
            self.glove_embs[i] = self.glove[self.tags[i]]
        self.glove_embs = torch.tensor(self.glove_embs.astype('float32'))
        if os.path.exists(valid_gt_fn):
            self.valid_gt = np.load(valid_gt_fn)
        else:
            print('get valid ground truth binaries')
            self.valid_gt = np.zeros((len(self.valid_list), len(self.tags)))
            i = 0
            for _song in tqdm.tqdm(self.valid_list):
                annot = list(track_tags[track_tags.id==_song].cleaned)
                for _tag in annot:
                    self.valid_gt[i][self.tags.index(_tag)] = 1
                i += 1
            np.save(open(valid_gt_fn, 'wb'), self.valid_gt)
            print('done!')

    def to_var(self, x):
        if self.is_cuda:
            x = x.cuda()
        return x

    def save(self, filename):
        model = self.model.state_dict()
        torch.save({'model': model}, os.path.join(self.model_save_path, filename))

    def load(self, filename):
        S = torch.load(filename)
        SS = {}
        for k, v in S['model'].items():
            SS[k] = v
        self.model.load_state_dict(SS)
        print('model %s loaded' % os.path.basename(filename)[:-4])

    def print_log(self, ctr, loss, epoch, start_t):
        length = len(self.data_loader)
        if ctr % self.log_step == 0:
            print("[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" %
                    (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                     epoch+1, self.n_epochs, ctr, length, np.sqrt(loss.item()),
                     datetime.timedelta(seconds=time.time()-start_t)))

    def train(self):
        loss_function = TripletLoss(self.margin)

        start_t = time.time()
        best_score = 0
        for epoch in range(self.n_epochs):
            ctr = 0
            self.model.train()
            losses = []
            for anchor, positive, negative in self.data_loader:
                # forward
                anchor_emb, positive_emb, negative_emb = self.model(self.to_var(anchor), self.to_var(positive), self.to_var(negative))
                loss = loss_function(anchor_emb, positive_emb, negative_emb)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

                # Log
                ctr += 1
                self.print_log(ctr, loss, epoch, start_t)
            # tensorboard
            loss = np.mean(losses)

            # validation
            self.model.eval()
            p_ks, roc_aucs, aps = self.validation()
            score = np.mean(roc_aucs)
            if score > best_score:
                print('best model!')
                best_score = score
                self.save('model_%d.pth' % epoch)
                print('saved the model')

    def validation(self):
        self.model.eval()
        # song to embedding
        song_embs = np.zeros((len(self.valid_list), 256))
        song_per_batch = 4
        num_iter = len(self.valid_list) // song_per_batch
        remainder = len(self.valid_list) % song_per_batch
        for _iter in tqdm.tqdm(range(num_iter)):
            indice = np.arange(song_per_batch) + (_iter * song_per_batch)
            song_embs[indice] = self.song_to_emb(indice, song_per_batch)
        if remainder > 0:
            indice = np.arange(len(self.valid_list) - remainder, len(self.valid_list))
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

        return p_ks, roc_aucs, aps

    def song_to_emb(self, indice, song_per_batch):
        batch_size = 8
        length = 256
        batch = np.zeros((batch_size * song_per_batch, 128, length))
        for i, ix in enumerate(indice):
            song_id = self.valid_list[ix]
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
            gt = self.valid_gt.T[i][sorted_ix]
            p_k = metrics.precision_score(gt, np.ones(k))
            p_ks.append(p_k)
        return p_ks

    def get_roc_auc(self, sim_scores):
        roc_aucs = metrics.roc_auc_score(self.valid_gt, sim_scores.T, average=None)
        return roc_aucs

    def get_ap(self, sim_scores):
        ap = metrics.average_precision_score(self.valid_gt, sim_scores.T, average=None)
        return ap

