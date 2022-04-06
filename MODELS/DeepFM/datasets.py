import os
import torch
from torch.utils.data import Dataset
from utils import neg_sample, join_attribute, feature_matrix

import pandas as pd
import numpy as np

import timeit
from tqdm import tqdm


class TestDataset(Dataset):
    def __init__(self, args, rating_df, attr_df):
        self.args = args
        self.attr_df = attr_df
        
        self.data = pd.read_csv('data/train/joined_rating_df.csv')
        print("_feature_matrix start")
        self.X, self.y = feature_matrix(self.data ,self.args.attr) # args.attr

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def get_users(self):
        return len(set(self.data.loc[:, 'user']))

    def get_items(self):
        return len(set(self.data.loc[:, 'item']))

    def get_attributes(self):
        return len(set(self.data.loc[:, self.args.attr])) # args.attr

    def __len__(self):
        return len(self.data)

class RatingDataset(Dataset):
    def __init__(self, args, rating_df, attr_df):
        """
        Args:
            args      : arguments
            rating_df : ["user", "item", "rating"] rating=1.0
            attr_df   : ["item", "genre", "writer"] if attr_df = genre_writer.csv
        """
        
        self.args = args
        self.rating_df = neg_sample(rating_df, self.args.negative_num) # args.negative num
        self.attr_df = attr_df
        
        # self.data = ["user", "item", "rating", "genre", "writer"]
        self.data = join_attribute(self.rating_df, self.attr_df) # args.attr
        self.X, self.y = feature_matrix(data=self.data, attr=["genre", "writer"]) # args.attr
        
        self.train_df = pd.read_csv("./data/train/rating.csv")
        self.genre_writer_df = pd.read_csv("./data/train/genre_writer.csv")
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def get_users(self):
        return len(set(self.train_df["user"]))

    def get_items(self):
        return len(set(self.train_df["item"]))

    def get_attributes1(self):
        return len(set(self.genre_writer_df[self.args.attr[0]])) # args.attr
    
    def get_attributes2(self):
        return len(set(self.genre_writer_df[self.args.attr[1]])) # args.attr

    def __len__(self):
        return len(self.data)

class InferenceDataset(Dataset):
    def __init__(self, args, rating_dir, attr_dir):
        self.args = args
        self.rating_dir = rating_dir # ratings.csv
        self.attr_dir = attr_dir     # genre_writer.csv
        self.data = None
        self.user_dict = None
        self.item_dict = None
        self.offsets = None
        self.X = None
        
        self.train_df = pd.read_csv("./data/train/rating.csv") # for n_user, n_items
        self.genre_writer_df = pd.read_csv("./data/train/genre_writer.csv") # for n_genre, n_writer
        
        
        self.setup()
        
    def setup(self):
        INFERENCE_SAMPLE_PATH = "./data/train/inference_sample.csv" 
        
        #-- Read DataFrame "ratings.csv", "genre_writer.csv"
        rating_df = pd.read_csv(self.rating_dir)
        attr_df = pd.read_csv(self.attr_dir, index_col=0)
        
        #-- Make User unseen item list (data)
        print(f"[DEBUG] Create inference samples (user unseen data)...")
        
        if os.path.isfile(INFERENCE_SAMPLE_PATH):
            print(f"[INFO] {INFERENCE_SAMPLE_PATH} file exists. Using this csv file...")
            data = pd.read_csv(INFERENCE_SAMPLE_PATH)
        else:
            data = self._inference_sample(rating_df)
            data = pd.DataFrame(data, columns=["user","item"])
            data.to_csv(INFERENCE_SAMPLE_PATH, columns=["user","item"], index=False)
            print(f"[INFO] {INFERENCE_SAMPLE_PATH} generated...")
    
        print(f"[DEBUG] len(data) = {len(data)}")
        print('[DEBUG] Merge attribute dataframe')
        
        #-- Merge <User unseen data, item_genre_writer>
        joined_rating_df = pd.merge(data, attr_df, left_on='item', right_on='item', how='inner')
        print('[DEBUG] Merge finished')
        
        users = list(set(joined_rating_df.loc[:,'user']))
        users.sort()
        items = list(set((joined_rating_df.loc[:, 'item'])))
        items.sort()

        if len(users)-1 != max(users):
            users_dict = {users[i]: i for i in range(len(users))}
            self.user_dict = {v:k for k,v in users_dict.items()}
            joined_rating_df['user']  = joined_rating_df['user'].map(lambda x : users_dict[x])
            users = list(set(joined_rating_df.loc[:,'user']))
                
        if len(items) - 1 != max(items):
            items_dict = {items[i]: i for i in range(len(items))}
            self.item_dict = {v:k for k,v in items_dict.items()}
            joined_rating_df['item'] = joined_rating_df['item'].map(lambda x : items_dict[x])
            items = list(set((joined_rating_df.loc[:, 'item'])))

        joined_rating_df = joined_rating_df.sort_values(by=['user'])
        joined_rating_df.reset_index(drop=True, inplace=True)
        
        self.data = joined_rating_df
        self.X = self._feature_matrix(self.args.attr)
        
    def _inference_sample(self, rating_df: pd.DataFrame):
        
        items = set(rating_df['item'])
        data = list()
        
        #-- make user unseen data list (user, unseen_item)
        for user in tqdm(rating_df["user"].unique()):
            user_seen_items = set(rating_df[rating_df["user"] == user]["item"])
            user_unseen_items = list(items - user_seen_items)
            data.extend([u, i] for u, i in zip([user], user_unseen_items))


        return data

    def _feature_matrix(self, attr='genre'):
        
        #-- feature matrix X, label tensor y 생성
        user_col = torch.tensor(self.data.loc[:, 'user'])
        item_col = torch.tensor(self.data.loc[:, 'item'])
        attr_col1 = torch.tensor(self.data.loc[:, attr[0]])
        attr_col2 = torch.tensor(self.data.loc[:, attr[1]])

        n_user = len(set(user_col))
        n_item = len(set(item_col))
        n_attr1 = len(set(attr_col1))

        offsets = [0, n_user, n_user + n_item, n_user + n_item + n_attr1]
        for col, offset in zip([user_col, item_col, attr_col1, attr_col2], offsets):
            col += offset

        X = torch.cat(
            tensors=[user_col.unsqueeze(1), item_col.unsqueeze(1), attr_col1.unsqueeze(1), attr_col2.unsqueeze(1)],
            dim=1)

        return X.long()

    def decode_offset(self, user_id: int, item_id: np.array) -> tuple:
        user_idx = user_id - self.offsets[0] #[B]
        item_idx_array = (item_id - self.offsets[1]).astype(int) #[B]
        #attr_idx = X[2] - self.offset[2]

        #user = self.user_dict[user_idx]
        #item = self.item_dict[item_idx]
        users = self.user_dict[user_idx]
        items = [self.item_dict[i] for i in item_idx_array]
        return users, items

    def __getitem__(self, index):
        return self.X[index]

    def get_users(self):
        return len(set(self.train_df["user"]))

    def get_items(self):
        return len(set(self.train_df["item"]))

    def get_attributes1(self):
        return len(set(self.genre_writer_df[self.args.attr[0]])) # args.attr
    
    def get_attributes2(self):
        return len(set(self.genre_writer_df[self.args.attr[1]])) # args.attr

    def __len__(self):
        return len(self.data)