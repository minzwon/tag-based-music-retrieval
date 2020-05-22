import os
import sys
import numpy as np
import pandas as pd
import random
import pickle
import tqdm
from torch.utils import data

class DataLoader(data.Dataset):
    def __init__(self, data_path):
        print('loading..')
        self.data_path = data_path
        self.d = pickle.load(open(os.path.join(data_path, 'cross_modal/cross_modal_dict.pkl'), 'rb'))
        self.tags = list(self.d.keys())
        print('loaded!')

    def load_spec(self, song_id):
        fn = os.path.join(self.data_path, 'spec_clean', song_id[2], song_id[3], song_id[4], song_id+'.npy')
        length = 256
        spec = np.load(fn)
        random_ix = int(np.floor(np.random.random(1) * (spec.shape[1] - length)))
        spec = spec[:, random_ix:random_ix+length]
        return spec

    def __getitem__(self, index):
        tag = self.tags[index % len(self.tags)]
        pos_song, neg_song, anchor_emb = random.sample(self.d[tag]['pos_train'], 1)[0], random.sample(self.d[tag]['neg_train'], 1)[0], self.d[tag]['emb']
        pos_spec = self.load_spec(pos_song).astype('float32')
        neg_spec = self.load_spec(neg_song).astype('float32')
        return anchor_emb, pos_spec, neg_spec

    def __len__(self):
        return 5000


def get_data_loader(config):
    data_loader = data.DataLoader(dataset=DataLoader(config.data_path),
                                    batch_size=config.batch_size, shuffle=True, drop_last=True)
    return data_loader
