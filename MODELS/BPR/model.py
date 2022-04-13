import torch
import torch.nn as nn


class BPR(nn.Module):
	def __init__(self, user_num, item_num, factor_num):
		super(BPR, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
		"""		
		self.embed_user = nn.Embedding(user_num, factor_num)
		self.embed_item = nn.Embedding(item_num, factor_num) 

		nn.init.normal_(self.embed_user.weight, std=0.01)
		nn.init.normal_(self.embed_item.weight, std=0.01)

	def forward(self, user, item_i, item_j):
		# print (f"Forwarding user: {user.shape}, item_i : {item_i.shape}, item_j : {item_j.shape}")
  
		user_e = self.embed_user(user) # [100 x 16]
		item_i_e = self.embed_item(item_i) # [100 x 16]
		item_j_e = self.embed_item(item_j) # [100 x 16]

		# print (user_e * item_i_e)
  
		prediction_i = (user_e * item_i_e).sum(dim=-1) # 100
		prediction_j = (user_e * item_j_e).sum(dim=-1) # 100
  
		# print (f"Prediction i result : {prediction_i.shape}")
		# print (prediction_i)

		# print (f"Prediction j result : {prediction_j.shape}")
		# print (prediction_j)
  
		return prediction_i, prediction_j # [ 100 ]
