import pandas as pd
import numpy as np
from tqdm import tqdm

train_ratings = pd.read_csv('./data/train/train_ratings.csv')
bad_users = pd.read_csv('./data/train/bad_user_id_02.csv', header=None)

all_users = set(train_ratings['user'].unique())
bad_users = set(bad_users.values.squeeze())
good_users = all_users - bad_users
good_users = sorted(list(good_users))

users = []
items = []
times = []

for user_id in tqdm(good_users):
    idx = train_ratings[train_ratings['user']==user_id].index
    for user, item, time in train_ratings.iloc[idx].values:
        users.append(user)
        items.append(item)
        times.append(time)

train_ratings_good_users = pd.DataFrame({'user': users, 'item':items,'time':times})
train_ratings_good_users.to_csv('./data/train/train_ratings_good_user.csv')

print('Save train_ratings_good_user.csv ....')
print('--------------------------------------')
print(f'All users: {len(all_users)}')
print(f'Good Users: {len(good_users)}')
print(f'Bad Users: {len(bad_users)}')

print(f'Before preprocessing train_ratings: {len(train_ratings)}')
print(f'After preprocessing train_ratings: {len(train_ratings_good_users)}')