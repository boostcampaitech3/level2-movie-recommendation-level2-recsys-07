import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import os

def neg_sample(rating_df, num_negative):
    #-- Negative instance 생성
    print(f"[INFO] Create Nagetive instances")
    
    items = set(rating_df.loc[:, 'item'])
    user_group_dfs = list(rating_df.groupby('user')['item'])
    
    user_neg_dfs = np.array([]).reshape(0, 3)

    for u, user_seen_list in tqdm(user_group_dfs):
        
        #-- User가 시청한 영화 집합
        user_seen_set = set(user_seen_list)
        
        #-- 시청한 영화를 제외한 num_negative개의 영화 선택
        i_user_neg_item = np.random.choice(list(items - user_seen_set), num_negative, replace=False)
        
        #-- negative sample item's rating = 0
        neg_users = np.full(num_negative, u)
        neg_ratings = np.zeros(num_negative)
        
        #-- user u 에 대한 negative sample 결과 생성 : ["neg_user", "neg_item", "neg_rate"]
        neg_results = np.vstack((neg_users, i_user_neg_item, neg_ratings)).T
        user_neg_dfs = np.vstack((user_neg_dfs, neg_results))
    
    neg_rating_df = pd.DataFrame(data=user_neg_dfs, columns=["user", "item", "rating"])
    rating_df = pd.concat([rating_df, neg_rating_df], axis=0, sort=False)
    
    return rating_df


def join_attribute(rating_df, attr_df):
    print('Join attribute df')
    
    #-- Join DataFrames on "item" column
    JOINED_RATING_PATH = "./data/train/joined_rating_df.csv"
    if os.path.isfile(JOINED_RATING_PATH):
        print(f"[INFO] Joined rating DataFrame exists. Using this csv file...")
        joined_rating_df = pd.read_csv(JOINED_RATING_PATH)

    else:
        joined_rating_df = pd.merge(rating_df, attr_df, left_on='item', right_on='item', how='inner')

        # user, item을 zero-based index로 mapping
        users = list(set(joined_rating_df.loc[:, 'user']))
        users.sort()
        items =  list(set((joined_rating_df.loc[:, 'item'])))
        items.sort()

        if len(users)-1 != max(users):
            users_dict = {users[i]: i for i in range(len(users))}
            joined_rating_df['user'] = joined_rating_df['user'].map(lambda x : users_dict[x])
            users = list(set(joined_rating_df.loc[:,'user']))
            
        if len(items)-1 != max(items):
            items_dict = {items[i]: i for i in range(len(items))}
            joined_rating_df['item'] = joined_rating_df['item'].map(lambda x : items_dict[x])
            items = list(set((joined_rating_df.loc[:, 'item'])))

        print (f"[DEBUG] Start sorting DataFrame by 'user' column")
        
        joined_rating_df = joined_rating_df.sort_values(by=['user'])
        joined_rating_df.reset_index(drop=True, inplace=True)
    
        #-- Save sorted DataFrame to csv file
        print (f"[INFO] {JOINED_RATING_PATH} does not exists. Making file...")
        joined_rating_df.to_csv(
            path_or_buf=JOINED_RATING_PATH,
            columns=["user", "item", "rating", "genre", "writer"],
            index=False)

    return joined_rating_df


def feature_matrix(data, attr='genre'):
    """
    feature matrix X, label tensor y 생성

    Args:
        data (DataFrame): Joined DataFrame (cols=["user", "item", "rating", "genre", "writer"])
        attr (str, optional): 사용할 attributes. Defaults to 'genre'.

    Returns:
        X, y : 입력과 정답
    """
    
    #-- user, item, attr1, attr2 의 column list 생성
    user_col = torch.tensor(data["user"])
    print (f"[DEBUG] user_col done.")
    item_col = torch.tensor(data["item"])
    print (f"[DEBUG] item_col done.")
    attr1_col = torch.tensor(data[attr[0]])
    print (f"[DEBUG] genre_col done.")
    attr2_col = torch.tensor(data[attr[1]])
    print (f"[DEBUG] writer_col done.")

    print (f"[DEBUG] Get Columns lists in feature_matrix function")
    
    #-- offset을 위한 각 attribute의 length 계산
    n_user = len(set(data.loc[:, 'user']))
    n_item = len(set(data.loc[:, 'item']))
    n_attr1 = len(set(data.loc[:, attr[0]]))
    
    print (f"[DEBUG] n_user : {n_user}, n_item : {n_item}, n_attr1 : {n_attr1}")
    
    #-- offset 설정
    offsets = [0, n_user, n_user + n_item, n_user + n_item + n_attr1]
    
    for col, offset in zip([user_col, item_col, attr1_col, attr2_col], offsets):
        col += offset

    #-- 모델의 입력과 정답
    X = torch.cat(
        tensors=[user_col.unsqueeze(1), item_col.unsqueeze(1), attr1_col.unsqueeze(1), attr2_col.unsqueeze(1)], 
        dim=1)
    y = torch.tensor(list(data.loc[:,'rating']))

    return X.long(), y.long()

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__