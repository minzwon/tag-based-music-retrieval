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

from model import AudioModel, CFModel, HybridModel

class Solver(object):
	def __init__(self, data_path, model_path, model_type, batch_size, w2v_type='music', is_cuda=True):
		self.data_path = data_path
		self.batch_size = batch_size
		self.model_type = model_type
		self.w2v_type = w2v_type
		self.is_cuda = is_cuda
		# load model
		self.model = self.load_model(model_path, model_type)
		if self.is_cuda:
			self.model = self.model.cuda()
		# load testset files
		self.load_test_files()

	def load_model(self, model_path, model_type):
		S = torch.load(model_path)['state_dict']
		SS = {key[6:]: S[key] for key in S.keys()}
		if model_type == 'spec':
			model = AudioModel()
		elif model_type == 'cf':
			model = CFModel()
		elif model_type == 'hybrid':
			model = HybridModel()
		model.load_state_dict(SS)
		return model

	def load_test_files(self):
		# load files
		self.test_ids = np.load(os.path.join(self.data_path, 'test_ids.npy'))
		ix_to_cf = np.load(os.path.join(self.data_path, 'ix_to_cf.npy'))
		self.tags = np.load(os.path.join(self.data_path, 'tags.npy'))
		binaries = np.load(os.path.join(self.data_path, 'binaries.npy'))
		self.word_emb = pickle.load(open(os.path.join(self.data_path, '%s_emb.pkl'%self.w2v_type), 'rb'))

		# preprocess
		indice = [int(line.split('//')[0]) for line in self.test_ids]
		self.ground_truth = binaries[indice]
		self.ix_to_cf = ix_to_cf[indice]

	def to_var(self, x):
		if self.is_cuda:
			x = x.cuda()
		return x

	def evaluation(self):
		self.model.eval()
		# song to embedding
		song_embs = self.song_to_emb()

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

	def song_to_emb(self):
		"""
			Need to be updated
		"""
		if self.model_type == 'spec':
			return 0

		elif self.model_type == 'cf':
			embs = []
			for i in tqdm.tqdm(range(len(self.ix_to_cf)//self.batch_size)):
				inp = self.to_var(torch.tensor(self.ix_to_cf[i * self.batch_size:(i+1) * self.batch_size]))
				out = self.model.cf_to_embedding(torch.tensor(inp)).detach().cpu()
				embs.append(out)
			inp = self.to_var(torch.tensor(self.ix_to_cf[(i+1) * self.batch_size:]))
			out = self.model.cf_to_embedding(torch.tensor(inp)).detach().cpu()
			embs.append(out)
			song_embs = torch.cat(embs, dim=0)
			return song_embs

		elif self.model_type == 'hybrid':
			return 0

		return song_embs

	def tags_to_emb(self):
		word_emb = torch.tensor([self.word_emb[key] for key in self.word_emb.keys()])
		tag_emb = self.model.word_to_embedding(self.to_var(word_emb)).cpu().detach()
		return tag_emb

	def get_similarity(self, tag_embs, song_embs):
		sim_scores = np.zeros((len(tag_embs), len(song_embs)))
		for i in range(50):
			sim_scores[i] = np.array(nn.CosineSimilarity(dim=-1)(tag_embs[i], song_embs))
		return sim_scores

	def get_precision(self, sim_scores, k=10):
		p_ks = []
		for i in range(50):
			sorted_ix = np.argsort(sim_scores[i])[::-1][:k]
			gt = self.ground_truth.T[i][sorted_ix]
			p_k = metrics.precision_score(gt, np.ones(k))
			p_ks.append(p_k)
		return p_ks

	def get_roc_auc(self, sim_scores):
		roc_aucs = metrics.roc_auc_score(self.ground_truth, sim_scores.T, average=None)
		return roc_aucs

	def get_ap(self, sim_scores):
		ap = metrics.average_precision_score(self.ground_truth, sim_scores.T, average=None)
		return ap

