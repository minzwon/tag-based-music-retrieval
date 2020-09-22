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
		self.num_workers = config.num_workers
		self.input_type = config.input_type
		self.w2v_type = config.w2v_type
		self.is_balanced = config.is_balanced
		self.is_weighted = config.is_weighted

		# load validation data
		self.load_valid_data(config.data_path, config.w2v_type)
		
		# triplet loss
		self.triplet_loss = TripletLoss(config.margin)

		# model
		if config.input_type=='spec':
			self.model = AudioModel()
		elif config.input_type=='cf':
			self.model = CFModel()
		elif config.input_type=='hybrid':
			self.model = HybridModel()

	def load_valid_data(self, data_path, w2v_type):
		# get w2v word embedding
		emb_dict = pickle.load(open(os.path.join(data_path, '%s_emb.pkl'%w2v_type), 'rb'))
		self.word_emb = torch.tensor([emb_dict[key] for key in emb_dict.keys()]).cuda()
		# get valid data
		self.valid_ids = np.load(os.path.join(data_path, 'valid_ids.npy'))
		self.tags = np.load(os.path.join(data_path, 'tags.npy'))

		# preprocess
		ix_to_cf = np.load(os.path.join(data_path, 'ix_to_cf.npy'))
		binaries = np.load(os.path.join(data_path, 'binaries.npy'))
		indice = [int(line.split('//')[0]) for line in self.valid_ids]
		self.ground_truth = binaries[indice]
		self.ix_to_cf = ix_to_cf[indice]

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
		return [optimizer], [scheduler]

	def train_loss(self, anchor, positive, negative):
		return self.triplet_loss(anchor, positive, negative)

	def valid_loss(self, tag_emb, song_emb):
		sims = nn.CosineSimilarity(dim=-1)(tag_emb, song_emb)
		return 1 - sims.mean()

	def train_dataloader(self):
		return DataLoader(dataset=MyDataset(self.data_path, split='TRAIN', input_type=self.input_type, 
											w2v_type=self.w2v_type, is_balanced=self.is_balanced), 
						  batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers)

	def val_dataloader(self):
		return DataLoader(dataset=MyDataset(self.data_path, split='VALID', input_type=self.input_type,
										    w2v_type=self.w2v_type, is_balanced=self.is_balanced),
						  batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)

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
		tag, spec, cf, _, _ = batch
		tag_emb, song_emb = self.model.forward(tag, spec, cf)
		loss = self.valid_loss(tag_emb, song_emb)
		logs = {"loss": loss}
		return {"val_loss": loss,
				"log": logs}

	def validation_epoch_end(self, outputs):
		p_10, roc_auc, ap = self.custom_validation()
		avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
		tensorboard_logs = {"val_loss": avg_loss,
							"p@10": p_10,
							"roc_auc": roc_auc,
							"map": ap}
		return {"avg_val_loss": avg_loss, 
				"p@10": p_10,
				"roc_auc": roc_auc,
				"map": ap,
				"log": tensorboard_logs}

	# functions for custom validation
	def custom_validation(self):
		tag_embs = self.tags_to_emb()
		song_embs = self.song_to_emb()

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

	def tags_to_emb(self):
		tag_emb = self.model.word_to_embedding(self.word_emb).detach().cpu()
		return tag_emb

	def song_to_emb(self):
		if self.input_type == 'spec':
		    return 0
		elif self.input_type == 'cf':
			embs = []
			for i in tqdm.tqdm(range(len(self.valid_ids)//self.batch_size)):
				inp = torch.tensor(self.ix_to_cf[i * self.batch_size:(i+1) * self.batch_size]).cuda()
				out = self.model.cf_to_embedding(inp).detach().cpu()
				embs.append(out)
			inp = torch.tensor(self.ix_to_cf[(i+1) * self.batch_size:]).cuda()
			out = self.model.cf_to_embedding(inp).detach().cpu()
			embs.append(out)
			song_embs = torch.cat(embs, dim=0)
			return song_embs
		elif self.input_type == 'hybrid':
			return 0

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

	# triplet sampling
	def triplet_sampling(self, tag_emb, song_emb, tag_binary, song_binary):
		num_batch = len(tag_emb)
		if self.is_weighted:
			tag_norm = tag_emb / tag_emb.norm(dim=1)[:, None]
			song_norm = song_emb / song_emb.norm(dim=1)[:, None]
			dot_sim = torch.matmul(tag_norm, song_norm.T)
			weights = (dot_sim + 1) / 2
			mask = 1 - torch.matmul(tag_binary, song_binary.T)
			masked_weights = weights * mask
			index_array = torch.arange(num_batch)
			negative_ix = [random.choices(index_array, weights=masked_weights[i], k=1)[0].item() for i in range(num_batch)]
		else:
			index_array = torch.arange(num_batch)
			negative_ix = [random.choice(index_array) for _ in range(num_batch)] 
		negative_emb = song_emb[negative_ix]
		return tag_emb, song_emb, negative_emb


