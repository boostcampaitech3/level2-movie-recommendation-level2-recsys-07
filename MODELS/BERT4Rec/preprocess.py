import pandas as pd
import numpy as np
import os

from collections import defaultdict
from pyparsing import col
from tqdm import tqdm

def preprocess(args, option):
    
    print ("[INFO] Pre-process train_ratings.csv (re-indexing)")
    
    #-- Arguments
    DATA_PATH = args.train_rating_path
    BATCH_SIZE = args.batch_size
    PREPROCESSED_PATH = "./data/preprocessed_ratings.csv"
    
    train_rating_df = None
    
    if os.path.isfile(PREPROCESSED_PATH):
        print(f"[INFO] {PREPROCESSED_PATH} file exists. Using this csv file...")
        train_rating_df = pd.read_csv(PREPROCESSED_PATH)
    
    else:
        #-- Read DataFrame (train_ratings.csv)
        train_rating_df = pd.read_csv(DATA_PATH)

        item_ids = train_rating_df['item'].unique() # 6807
        user_ids = train_rating_df['user'].unique() # 31360

        #-- Re-index user, item
        # CAUTION : user starts with index 0 (0 ~ 31359)
        #           item starts with index 1 (1 ~ 6807) 
        item2idx = pd.Series(data=np.arange(len(item_ids)) + 1, index=item_ids)
        user2idx = pd.Series(data=np.arange(len(user_ids)),     index=user_ids)

        #-- Make DataFrame with re-indexed user & item ["timestamp", "user_reidx", "item_reidx"]
        train_rating_df = pd.merge(train_rating_df, pd.DataFrame({'item': item_ids, 'item_reidx': item2idx[item_ids].values}), on='item', how='inner')
        train_rating_df = pd.merge(train_rating_df, pd.DataFrame({'user': user_ids, 'user_reidx': user2idx[user_ids].values}), on='user', how='inner')
        train_rating_df.sort_values(['user_reidx', 'time'], inplace=True)
        del train_rating_df['item'], train_rating_df['user'] 

        #-- Save preprocessed DataFrame
        # [user, item, timestamp] -> [timestamp, user_reindex, item_reindex]
        train_rating_df.to_csv(PREPROCESSED_PATH, columns=["time", "user_reidx", "item_reidx"], index=False)


    #-- Make user's seen movie list
    # [user1] : [item1, item2, item5, ... , item78]
    print (f"[INFO] Make user's seen movie list")
    user_seen_dict = defaultdict(list)
    for u, i in tqdm(zip(train_rating_df['user_reidx'], train_rating_df['item_reidx']), total=len(train_rating_df['user_reidx'])):
        user_seen_dict[u].append(i)


    # train set, valid set 생성
    user_train = {}
    user_valid = {}
    
    last_idx = None
    if option == "leave_one_out":
        last_idx = -2
    elif option == "split_by_user":
        last_idx = -1
    
    for user in user_seen_dict:
        user_train[user] =  user_seen_dict[user][:last_idx]
        user_valid[user] = [user_seen_dict[user][last_idx]]


    num_user = len(train_rating_df["user_reidx"].unique()) # 31360
    num_item = len(train_rating_df["item_reidx"].unique()) # 6807
    print(f'\n[INFO] num users: {num_user}, num items: {num_item}')
    
    return num_user, num_item, train_rating_df, user_train, user_valid