import numpy as np
import torch
import torch.nn as nn


class Conv_2d(nn.Module):
	def __init__(self, input_channels, output_channels, shape=3, pooling=2):
		super(Conv_2d, self).__init__()
		self.conv = nn.Conv2d(input_channels, output_channels, shape, padding=shape//2)
		self.bn = nn.BatchNorm2d(output_channels)
		self.relu = nn.ReLU()
		self.mp = nn.MaxPool2d(pooling)

	def forward(self, x):
		out = self.mp(self.relu(self.bn(self.conv(x))))
		return out


class Conv_emb(nn.Module):
	def __init__(self, input_channels, output_channels):
		super(Conv_emb, self).__init__()
		self.conv = nn.Conv2d(input_channels, output_channels, 1)
		self.bn = nn.BatchNorm2d(output_channels)
		self.relu = nn.ReLU()

	def forward(self, x):
		out = self.relu(self.bn(self.conv(x)))
		return out

