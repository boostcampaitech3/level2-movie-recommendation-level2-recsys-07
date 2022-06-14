import numpy as np 
import pandas as pd 
import scipy.sparse as sp

import torch.utils.data as data

import config
import json
import tqdm

def load_all(test_num=100):
	""" We load all the three file here to save time in each epoch. """
	train_data = pd.read_csv(
		config.train_rating, header=None, names=['user', 'item'], 
		usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

	user_num = train_data['user'].max() + 1
	item_num = train_data['item'].max() + 1

	train_data = train_data.values.tolist()

	# load ratings as a dok matrix
	print ("load ratings as a dok matrix")
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in tqdm.tqdm(train_data):
		train_mat[x[0], x[1]] = 1.0
  

	test_data = []
	print ("loading negative samples")
	with open(config.test_negative, "r") as fd:
		negatives = json.load(fd)
		for user in tqdm.tqdm(negatives.keys()):
			for neg_item in negatives[user]:
				test_data.append([int(user), int(neg_item)])
 
	print ("Dataset all loaded!")
	return train_data, test_data, user_num, item_num, train_mat


class BPRData(data.Dataset):
	def __init__(self, features, 
				num_item, train_mat=None, num_ng=0, is_training=None):
		super(BPRData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features = features
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training

	def ng_sample(self):
		assert self.is_training, 'no need to sampling when testing'

		self.features_fill = []
		for x in tqdm.tqdm(self.features):
			u, i = x[0], x[1]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while (u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_fill.append([u, i, j])

	def __len__(self):
		return self.num_ng * len(self.features) if \
				self.is_training else len(self.features)

	def __getitem__(self, idx):
		features = self.features_fill if \
				self.is_training else self.features

		user = features[idx][0]
		item_i = features[idx][1]
		item_j = features[idx][2] if \
				self.is_training else features[idx][1]
		return user, item_i, item_j 
		