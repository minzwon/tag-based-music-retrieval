import os
import random
import torch
import time
import pickle
import tqdm
import numpy as np
from sklearn import metrics
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule

from data_loader import MyDataset
from model import AudioModel, CFModel, HybridModel


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


class Solver(LightningModule):
	def __init__(self, config):
		super().__init__()
		# configuration
		self.lr = config.lr
		self.data_path = config.data_path
		self.batch_size = config.batch_size
		self.num_chunk = config.num_chunk
		self.input_length = config.input_length
		self.num_workers = config.num_workers
		self.input_type = config.input_type
		self.w2v_type = config.w2v_type
		self.is_balanced = config.is_balanced
		self.is_weighted = config.is_weighted
		self.is_subset = config.is_subset
		if config.is_subset:
			self.prefix = 'sub_'
		else:
			self.prefix = ''

		# load validation data
		self.load_eval_data(config.data_path, config.w2v_type, config.mode)
		
		# triplet loss
		self.triplet_loss = TripletLoss(config.margin)

		# model
		if config.input_type=='spec':
			self.model = AudioModel()
		elif config.input_type=='cf':
			self.model = CFModel()
		elif config.input_type=='hybrid':
			self.model = HybridModel()

		self.song_embs = []

	def load_eval_data(self, data_path, w2v_type, mode):
		# get w2v word embedding
		emb_dict = pickle.load(open(os.path.join(data_path, self.prefix+'%s_emb.pkl'%w2v_type), 'rb'))
		self.word_emb = torch.tensor([emb_dict[key] for key in emb_dict.keys()]).cuda()
		# get valid data
		if mode == 'TRAIN':
			self.eval_ids = np.load(os.path.join(data_path, self.prefix+'valid_ids.npy'))
		elif mode == 'TEST':
			self.eval_ids = np.load(os.path.join(data_path, self.prefix+'test_ids.npy'))
		self.tags = np.load(os.path.join(data_path, self.prefix+'tags.npy'))

		# preprocess
		binaries = np.load(os.path.join(data_path, self.prefix+'binaries.npy'))
		indice = [int(line.split('//')[0]) for line in self.eval_ids]
		self.ground_truth = binaries[indice]
		if self.input_type != 'spec':
			ix_to_cf = np.load(os.path.join(data_path, self.prefix+'ix_to_cf.npy'))
			self.ix_to_cf = ix_to_cf[indice]

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
		return optimizer

	def train_loss(self, anchor, positive, negative):
		return self.triplet_loss(anchor, positive, negative)

	def valid_loss(self, tag_emb, song_emb):
		sims = nn.CosineSimilarity(dim=-1)(tag_emb, song_emb)
		return 1 - sims.mean()

	def train_dataloader(self):
		return DataLoader(dataset=MyDataset(self.data_path, split='TRAIN', input_type=self.input_type, 
											input_length=self.input_length, w2v_type=self.w2v_type, 
											is_balanced=self.is_balanced, is_subset=self.is_subset), 
						  batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers)

	def val_dataloader(self):
		return DataLoader(dataset=MyDataset(self.data_path, split='VALID', input_type=self.input_type,
											input_length=self.input_length, num_chunk=self.num_chunk,
											w2v_type=self.w2v_type, is_balanced=self.is_balanced, is_subset=self.is_subset),
						  batch_size=self.batch_size//self.num_chunk, shuffle=False, drop_last=False, num_workers=self.num_workers)

	def test_dataloader(self):
		return DataLoader(dataset=MyDataset(self.data_path, split='TEST', input_type=self.input_type,
											input_length=self.input_length, num_chunk=self.num_chunk,
											w2v_type=self.w2v_type, is_balanced=self.is_balanced, is_subset=self.is_subset),
						  batch_size=self.batch_size//self.num_chunk, shuffle=False, drop_last=False, num_workers=self.num_workers)

	def training_step(self, batch, batch_idx):
		tag, spec, cf, tag_binary, song_binary = batch
		tag_emb, song_emb = self.model.forward(tag, spec, cf)
		anchor, positive, negative = self.triplet_sampling(tag_emb, song_emb, song_binary, tag_binary)
		loss = self.train_loss(anchor, positive, negative)
		logs = {"loss": loss}
		return {"loss": loss, "log": logs}

	def training_epoch_end(self, outputs):
		avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
		return {"loss": avg_loss}

	def validation_step(self, batch, batch_idx):
		_, spec, cf, _, _ = batch
		song_emb = self.song_to_emb(spec, cf)
		self.song_embs.append(song_emb)

	def validation_epoch_end(self, outputs):
		tag_embs = self.tags_to_emb()
		song_embs = torch.cat(self.song_embs, dim=0)

		# ignore unit test
		if song_embs.size(0) < self.batch_size:
			p_10, roc_auc, ap = torch.tensor(0), torch.tensor(0), torch.tensor(0)
		else:
			p_10, roc_auc, ap = self.get_scores(tag_embs, song_embs)
		tensorboard_logs = {"p@10": p_10,
							"roc_auc": roc_auc,
							"map": ap}
		self.song_embs = []
		return {"p@10": p_10,
				"roc_auc": roc_auc,
				"map": ap,
				"log": tensorboard_logs}

	def test_step(self, batch, batch_idx):
		_, spec, cf, _, _ = batch
		song_emb = self.song_to_emb(spec, cf)
		self.song_embs.append(song_emb)

	def test_epoch_end(self, outputs):
		tag_embs = self.tags_to_emb()
		song_embs = torch.cat(self.song_embs, dim=0)

		# ignore unit test
		if song_embs.size(0) < self.batch_size:
			p_10, roc_auc, ap = torch.tensor(0), torch.tensor(0), torch.tensor(0)
		else:
			p_10, roc_auc, ap = self.get_scores(tag_embs, song_embs)
		tensorboard_logs = {"p@10": p_10,
							"roc_auc": roc_auc,
							"map": ap}
		self.song_embs = []
		return {"p@10": p_10,
				"roc_auc": roc_auc,
				"map": ap,
				"log": tensorboard_logs}
				
	def tags_to_emb(self):
		tag_emb = self.model.word_to_embedding(self.word_emb).detach().cpu()
		return tag_emb

	def song_to_emb(self, spec, cf):
		if self.input_type == 'spec':
			b, c, f, t = spec.size()
			out = self.model.spec_to_embedding(spec.view(-1, f, t))
			out = out.view(b, c, -1)
			song_emb = out.mean(dim=1).detach().cpu()
		elif self.input_type == 'cf':
			song_emb = self.model.cf_to_embedding(cf).detach().cpu()
		elif self.input_type == 'hybrid':
			b, c, f, t = spec.size()
			out = self.model.spec_to_embedding(spec.view(-1, f, t))
			out = out.view(b, c, -1)
			spec_emb = out.mean(dim=1)
#			cf_emb = self.model.cf_to_embedding(cf)
			cat_emb = torch.cat([spec_emb, cf], dim=-1)
			song_emb = self.model.cat_to_embedding(cat_emb).detach().cpu()
		return song_emb

	# evaluation metrics
	def get_scores(self, tag_embs, song_embs):
		# get similarity score (tag x song)
		sim_scores = self.get_similarity(tag_embs, song_embs)

		# get metrics
		k = 10
		p_ks = self.get_precision(sim_scores, k=k)
		roc_aucs = self.get_roc_auc(sim_scores)
		aps = self.get_ap(sim_scores)

		# print
		print('precision @%d: %.4f' % (k, np.mean(p_ks)))
		print('roc_auc: %.4f' % np.mean(roc_aucs))
		print('map: %.4f' % np.mean(aps))
		for i, tag in enumerate(self.tags):
			print('%s: %.1f, %.4f, %.4f' % (tag, p_ks[i], roc_aucs[i], aps[i]))
		return torch.tensor(np.mean(p_ks)), torch.tensor(np.mean(roc_aucs)), torch.tensor(np.mean(aps))

	def get_similarity(self, tag_embs, song_embs):
		sim_scores = np.zeros((len(tag_embs), len(song_embs)))
		for i in range(len(tag_embs)):
			sim_scores[i] = np.array(nn.CosineSimilarity(dim=-1)(tag_embs[i], song_embs))
		return sim_scores

	def get_precision(self, sim_scores, k=10):
		p_ks = []
		for i in range(len(sim_scores)):
			sorted_ix = np.argsort(sim_scores[i])[::-1][:k]
			gt = self.ground_truth.T[i][sorted_ix]
			p_k = metrics.precision_score(gt, np.ones(k))
			p_ks.append(p_k)
		return p_ks

	def get_roc_auc(self, sim_scores):
		return metrics.roc_auc_score(self.ground_truth, sim_scores.T, average=None)

	def get_ap(self, sim_scores):
		return metrics.average_precision_score(self.ground_truth, sim_scores.T, average=None)

	def triplet_sampling(self, tag_emb, song_emb, tag_binary, song_binary):
		num_batch = len(tag_emb)
		if self.is_weighted:
			# get distance weights
			tag_norm = tag_emb / tag_emb.norm(dim=1)[:, None]
			song_norm = song_emb / song_emb.norm(dim=1)[:, None]
			dot_sim = torch.matmul(tag_norm, song_norm.T)
			weights = (dot_sim + 1) / 2

			# masking
			mask = 1 - torch.matmul(tag_binary, song_binary.T)
			masked_weights = weights * mask

			# sampling
			index_array = torch.arange(num_batch)
			negative_ix = [random.choices(index_array, weights=masked_weights[i], k=1)[0].item() for i in range(num_batch)]
		else:
			# masking
			mask = 1 - torch.matmul(tag_binary, song_binary.T)

			# sampling
			index_array = torch.arange(num_batch)
			negative_ix = [random.choices(index_array, weights=mask[i], k=1)[0].item() for i in range(num_batch)] 
		negative_emb = song_emb[negative_ix]
		return tag_emb, song_emb, negative_emb

