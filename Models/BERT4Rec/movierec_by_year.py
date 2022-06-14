import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import torch

print('Load rating.csv data...')
train_df = pd.read_csv("/opt/ml/input/data/train/train_ratings.csv")
conv_year = list()
for t in train_df['time']:
    y, m = map(int, time.strftime('%Y-%m', time.localtime(t)).split("-"))
    conv_year.append(y)
train_df['year'] = conv_year
train_df.drop(['time'], axis=1, inplace=True)
train_np = train_df.to_numpy()
# train_df = torch.tensor(train_df.values)


print('Load years.tsv data...')
movie_info = pd.read_csv('/opt/ml/input/data/train/years.tsv', sep='\t')
movie_info_np = movie_info.to_numpy()
# movie_info = torch.tensor(movie_info.values)


uniq_user = np.unique(train_np[:, 0])
movie = np.unique(train_np[:, 1])
answer_list = np.zeros((1, 2), dtype=np.int64)
original_sample = 0
new_sample = 0

print('Calculating sampling...')
for user in tqdm(uniq_user):
    '''
    train_np : [user, item, year]
    movie_info_np : [item, year]
    '''
    movie_seen = set((train_np[train_np[:,0] == user][:,1])) # item
    user_time = set((train_np[train_np[:,0] == user][:,2]))  # year
    max_year = max(user_time) + 1 # 끝 년도

    movie_unseen = set(movie) - movie_seen
    movie_in_user_time = set(movie_info_np[movie_info_np[:,1] <= max_year][:,0]) # item
    
    sample = movie_unseen & movie_in_user_time

    original_sample += len(movie_unseen)
    new_sample += len(sample)

    sample = np.array(list(sample))[:, np.newaxis]
    user_id = np.array([[user]]*len(sample))

    # print(sample.shape)
    # print(user_id.shape)

    sample = np.concatenate((user_id, sample), axis=1)
    answer_list = np.concatenate((answer_list, sample), axis=0)

#print(answer_list)
print(answer_list.shape)

answer_list = pd.DataFrame(answer_list, columns=['user', 'item'])
answer_list = answer_list.drop(0, axis=0).reset_index(drop=True)
answer_list.to_csv('./neg_sample_by_year.csv', index=False)
print('Done!!!')

print(f'Original neg sample: {original_sample}')
print(f'After sampling:{new_sample}')
print(f'diff: {original_sample-new_sample}')