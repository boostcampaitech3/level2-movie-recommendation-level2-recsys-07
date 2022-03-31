import torch
from torch.utils.data import Dataset
from utils import neg_sample, join_attribute, feature_matrix


class RatingDataset(Dataset):
    def __init__(self, args, rating_df, attr_df):
        self.args = args
        self.rating_df = neg_sample(rating_df, 50) # args.negative num
        self.attr_df = attr_df
        self.data = join_attribute(self.rating_df, self.attr_df, 'genre') # args.attr
        self.X, self.y = feature_matrix(self.data, 'genre') # args.attr

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def get_users(self):
        return len(set(self.data.loc[:, 'user']))

    def get_items(self):
        return len(set(self.data.loc[:, 'item']))

    def get_attributes(self):
        return len(set(self.data.loc[:, 'genre'])) # args.attr

    def __len__(self):
        return len(self.data)