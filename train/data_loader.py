import os
import sys
import pickle
import tqdm
import numpy as np
import pandas as pd
import random
from torch.utils import data


class MyDataset(data.Dataset):
	def __init__(self, data_path, split='TRAIN', input_type='spec', input_length=None, num_chunk=16, w2v_type='google', is_balanced=True):
		self.data_path = data_path
		self.split = split
		self.input_type = input_type
		self.input_length = input_length
		self.num_chunk = num_chunk
		self.is_balanced = is_balanced
		self.w2v_type = w2v_type

		# load ids
		if split == 'TRAIN':
			self.tag_to_ix = pickle.load(open(os.path.join(data_path, 'train_tag_to_ix.pkl'), 'rb'))
			self.train_ids = np.load(os.path.join(data_path, 'train_ids.npy'))
			self.get_tag_binaries()
		elif split == 'VALID':
			self.eval_ids = np.load(os.path.join(data_path, 'valid_ids.npy'))
		elif split == 'TEST':
			self.eval_ids = np.load(os.path.join(data_path, 'test_ids.npy'))

		# load binaries
		self.ix_to_binary = np.load(os.path.join(data_path, 'binaries.npy'))

		# load tag embedding
		self.load_tag_emb()

		# load collaborative filtering embedding
		if input_type != 'spec':
			self.cf = np.load(os.path.join(data_path, 'ix_to_cf.npy'))

	def get_tag_binaries(self):
		eye = np.eye(len(self.tag_to_ix))
		self.tag_binaries = {}
		for i, tag in enumerate(self.tag_to_ix.keys()):
			self.tag_binaries[tag] = eye[i]

	def load_tag_emb(self):
		self.tags = np.load(os.path.join(self.data_path, 'tags.npy'))
		self.w2v = pickle.load(open(os.path.join(self.data_path, '%s_emb.pkl'%self.w2v_type), 'rb'))

	def load_cf(self, song_ix):
		return self.cf[song_ix]

	def load_spec(self, song_id):
		fn = os.path.join(self.data_path, 'spec', song_id[2], song_id[3], song_id[4], song_id+'.npy')
		length = self.input_length
		spec = np.load(fn)

		# for short spectrograms
		if spec.shape[1] < self.input_length:
			nspec = np.zeros((128, self.input_length))
			nspec[:, :spec.shape[1]] = spec
			spec = nspec

		# multiple chunks for validation loader
		if self.split == 'TRAIN':
			time_ix = int(np.floor(np.random.random(1) * (spec.shape[1] - length)))
			spec = spec[:, time_ix:time_ix+length]
		elif (self.split == 'VALID') or (self.split == 'TEST'):
			hop = (spec.shape[1] - self.input_length) // self.num_chunk
			spec = np.array([spec[:, i*hop:i*hop+self.input_length] for i in range(self.num_chunk)])
		return spec

	def load_hybrid(self, song_ix, song_id):
		spec = self.load_spec(song_id)
		cf = self.load_cf(song_ix)
		return spec, cf

	def get_train_item(self, index):
		# song embedding
		if self.is_balanced:
			tag = self.tags[index % len(self.tags)]
			song_ix, song_id = random.choice(self.tag_to_ix[tag]).split('//')
			song_ix = int(song_ix)
			song_binary = self.ix_to_binary[song_ix]
		else:
			song_ix, song_id = random.choice(self.train_ids).split('//')
			song_ix = int(song_ix)
			song_binary = self.ix_to_binary[song_ix]
			tag = random.choices(self.tags, weights=song_binary, k=1)[0]

		# tag embedding
		tag_binary = self.tag_binaries[tag]
		tag_emb = self.w2v[tag]

		if self.input_type == 'spec':
			spec = self.load_spec(song_id)
			cf = np.array([])
		elif self.input_type == 'cf':
			spec = np.array([])
			cf = self.load_cf(song_ix)
		elif self.input_type == 'hybrid':
			spec, cf = self.load_hybrid(song_ix, song_id)
		return tag_emb, spec, cf, tag_binary, song_binary

	def get_eval_item(self, index):
		song_ix, song_id = self.eval_ids[index].split('//')
		song_ix = int(song_ix)
		tag_emb = np.array([])
		if self.input_type == 'spec':
			spec = self.load_spec(song_id)
			cf = np.array([])
		elif self.input_type == 'cf':
			spec = np.array([])
			cf = self.load_cf(song_ix)
		elif self.input_type == 'hybrid':
			spec, cf = self.load_hybrid(song_ix, song_id)
		song_binary = np.array([])
		tag_binary = np.array([])
		return tag_emb, spec, cf, tag_binary, song_binary

	def __getitem__(self, index):
		if self.split == 'TRAIN':
			tag_emb, spec, cf, tag_binary, song_binary = self.get_train_item(index)
		elif (self.split == 'VALID') or (self.split == 'TEST'):
			tag_emb, spec, cf, tag_binary, song_binary = self.get_eval_item(index)
		return tag_emb.astype('float32'), spec.astype('float32'), cf.astype('float32'), tag_binary, song_binary

	def __len__(self):
		if self.split == 'TRAIN':
			return 10000
		elif (self.split == 'VALID') or (self.split == 'TEST'):
			return len(self.eval_ids)


