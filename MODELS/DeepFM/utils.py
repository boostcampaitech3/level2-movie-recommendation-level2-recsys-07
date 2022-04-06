import pandas as pd
from tqdm import tqdm
import numpy as np
import torch

def neg_sample(rating_df, num_negative):
    # Negative instance 생성
    print("Create Nagetive instances")
    items = set(rating_df.loc[:, 'item'])
    num_negative = num_negative
    user_group_dfs = list(rating_df.groupby('user')['item'])
    first_row = True
    user_neg_dfs = pd.DataFrame()

    for u, u_items in tqdm(user_group_dfs):
        u_items = set(u_items)
        i_user_neg_item = np.random.choice(list(items - u_items), num_negative, replace=False)
        
        i_user_neg_df = pd.DataFrame({'user': [u]*num_negative, 'item': i_user_neg_item, 'rating': [0]*num_negative})
        if first_row == True:
            user_neg_dfs = i_user_neg_df
            first_row = False
        else:
            user_neg_dfs = pd.concat([user_neg_dfs, i_user_neg_df], axis = 0, sort=False)
    
    rating_df = pd.concat([rating_df, user_neg_dfs], axis = 0, sort=False)
    return rating_df


def join_attribute(rating_df, attr_df, attr='genre'):
    print('Join attribute df')
    # Join dfs
    joined_rating_df = pd.merge(rating_df, attr_df, left_on='item', right_on='item', how='inner')

    # user, item을 zero-based index로 mapping
    users = list(set(joined_rating_df.loc[:,'user']))
    users.sort()
    items =  list(set((joined_rating_df.loc[:, 'item'])))
    items.sort()
    attrs =  list(set((joined_rating_df.loc[:, attr])))
    attrs.sort()

    if len(users)-1 != max(users):
        users_dict = {users[i]: i for i in range(len(users))}
        joined_rating_df['user']  = joined_rating_df['user'].map(lambda x : users_dict[x])
        users = list(set(joined_rating_df.loc[:,'user']))
        
    if len(items)-1 != max(items):
        items_dict = {items[i]: i for i in range(len(items))}
        joined_rating_df['item']  = joined_rating_df['item'].map(lambda x : items_dict[x])
        items =  list(set((joined_rating_df.loc[:, 'item'])))

    joined_rating_df = joined_rating_df.sort_values(by=['user'])
    joined_rating_df.reset_index(drop=True, inplace=True)
    joined_rating_df.to_csv('./joined_rating_df.csv',index=False)

    return joined_rating_df


def feature_matrix(data, attr='genre'):
    #feature matrix X, label tensor y 생성
    user_col = torch.tensor(data.loc[:,'user'])
    item_col = torch.tensor(data.loc[:,'item'])
    attr_col = torch.tensor(data.loc[:,attr])

    n_user = len(set(data.loc[:,'user']))
    n_item = len(set(data.loc[:,'item']))

    offsets = [0, n_user, n_user+n_item]
    for col, offset in zip([user_col, item_col, attr_col], offsets):
        col += offset

    X = torch.cat([user_col.unsqueeze(1), item_col.unsqueeze(1), attr_col.unsqueeze(1)], dim=1)
    y = torch.tensor(list(data.loc[:,'rating']))

    return X.long(), y.long()


def get_unpopular_item(rating_df):
    df = rating_df.pivot_table(index=['item'], aggfunc='size').sort_values(ascending=True)
    df = df[df < 200]
    ret = df.index.to_numpy()
    return set(ret.flatten())

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__