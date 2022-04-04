import torch
from torch.utils.data import Dataset
from utils import neg_sample, join_attribute, feature_matrix

import pandas as pd
import numpy as np

from tqdm import tqdm


class TestDataset(Dataset):
    def __init__(self, args, rating_df, attr_df):
        self.args = args
        #self.rating_df = neg_sample(rating_df, self.args.negative_num) # args.negative num
        self.attr_df = attr_df

        self.data = pd.read_csv('data/train/joined_rating_df.csv')

        self.X, self.y = feature_matrix(self.data, self.args.attr) # args.attr

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
        self.args = args
        self.rating_df = neg_sample(rating_df, self.args.negative_num) # args.negative num
        self.attr_df = attr_df
        self.data = join_attribute(self.rating_df, self.attr_df, self.args.attr) # args.attr
        self.X, self.y = feature_matrix(self.data, self.args.attr) # args.attr

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

class InferenceDataset(Dataset):
    def __init__(self, args, rating_dir, attr_dir):
        self.args = args
        self.rating_dir = rating_dir
        self.attr_dir = attr_dir
        self.data = None
        self.user_dict = None
        self.item_dict = None
        self.offsets = None
        self.X = None
        
        self.setup()
        
    def setup(self):
        rating_df = pd.read_csv(self.rating_dir)
        attr_df = pd.read_csv(self.attr_dir, index_col = 0)
        
        print("Create inference samples")
        data = self._inference_sample(rating_df)
        data = pd.DataFrame(data, columns=["user","item"])
    
        print(f"# data length : {len(data)}")
        
        print('# Merge attribute dataframe')
        
        joined_rating_df = pd.merge(data, attr_df, left_on='item', right_on='item', how='inner')
        print('# Merge finished')
        users = list(set(joined_rating_df.loc[:,'user']))
        users.sort()
        items =  list(set((joined_rating_df.loc[:, 'item'])))
        items.sort()
        #attrs =  list(set((joined_rating_df.loc[:, "genre"])))
        #attrs.sort()

        if len(users)-1 != max(users):
            users_dict = {users[i]: i for i in range(len(users))}

            self.user_dict = {v:k for k,v in users_dict.items()}

            joined_rating_df['user']  = joined_rating_df['user'].map(lambda x : users_dict[x])
            users = list(set(joined_rating_df.loc[:,'user']))
                
        if len(items)-1 != max(items):
            items_dict = {items[i]: i for i in range(len(items))}
            
            self.item_dict = {v:k for k,v in items_dict.items()}

            joined_rating_df['item']  = joined_rating_df['item'].map(lambda x : items_dict[x])
            items =  list(set((joined_rating_df.loc[:, 'item'])))

        joined_rating_df = joined_rating_df.sort_values(by=['user'])
        joined_rating_df.reset_index(drop=True, inplace=True)
        
        #joined_rating_df.to_csv('data/train/inference_join.csv')

        self.data = joined_rating_df
        
        self.X = self._feature_matrix(self.args.attr)

    def _inference_sample(self,rating_df):
        items = set(rating_df['item'])
        user_rating = rating_df.groupby('user')['item'].apply(list)

        data = []

        for user, u_items in tqdm(user_rating.iteritems()):
            un_watched = [i for i in items if i not in u_items]
            data += [[user,i] for i in un_watched]
        return data

    def _feature_matrix(self, attr='genre'):
        #feature matrix X, label tensor y 생성
        user_col = torch.tensor(self.data.loc[:,'user'])
        item_col = torch.tensor(self.data.loc[:,'item'])
        attr_col = torch.tensor(self.data.loc[:,attr])

        n_user = len(set(self.data.loc[:,'user']))
        n_item = len(set(self.data.loc[:,'item']))

        self.offsets = [0, n_user, n_user+n_item]
        for col, offset in zip([user_col, item_col, attr_col], self.offsets):
            col += offset

        X = torch.cat([user_col.unsqueeze(1), item_col.unsqueeze(1), attr_col.unsqueeze(1)], dim=1)

        return X.long()

    def decode_offset(self, user_id: int, item_id: np.array) -> tuple(int,list):
        user_idx = user_id - self.offsets[0] #[B]
        item_idx_array = item_id - self.offsets[1] #[B]
        #attr_idx = X[2] - self.offset[2]

        #user = self.user_dict[user_idx]
        #item = self.item_dict[item_idx]
        users = self.user_dict[user_idx]
        items = [self.item_dict[i] for i in item_idx_array]
        return users, items

    def __getitem__(self, index):
        return self.X[index]

    def get_users(self):
        return len(set(self.data.loc[:, 'user']))

    def get_items(self):
        return len(set(self.data.loc[:, 'item']))

    def get_attributes(self):
        return len(set(self.data.loc[:, self.args.attr]))

    def __len__(self):
        return len(self.data)