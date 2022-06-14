import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import torch

print('Load rating.csv data...')
rating_df = pd.read_csv('/opt/ml/workspace/DeepFM/data/train/rating.csv')
rating_df = rating_df.pivot_table(index=['item'], aggfunc='size').sort_values(ascending=True)
rating_df = rating_df[rating_df < 200]
unpopular_movie = set(rating_df.index)

print('Load neg_sample_by_year.csv data...')
df = pd.read_csv('neg_sample_by_year.csv')
uniq_user = np.unique(df['user'])

answer_list = np.zeros((1, 2), dtype=np.int64)
original_sample = 0
new_sample = 0

print('Calculating sampling...')
for user in tqdm(uniq_user):
    '''
    df : [user, item]
    rating_df : [user, item, rating]
    '''
    sample = set(df[df['user']==user]['item'])
    original_sample += len(sample)

    sample = sample - unpopular_movie
    new_sample += len(sample)

    sample = np.array(list(sample))[:, np.newaxis]
    user_id = np.array([[user]]*len(sample))

    # print(sample.shape)
    # print(user_id.shape)

    sample = np.concatenate((user_id, sample), axis=1)
    answer_list = np.concatenate((answer_list, sample), axis=0)

print(answer_list)
print(answer_list.shape)

answer_list = pd.DataFrame(answer_list, columns=['user', 'item'])
answer_list = answer_list.drop(0, axis=0).reset_index(drop=True)
answer_list.to_csv('./neg_sample_final.csv', index=False)

print('Done!!')
print(f'Original neg sample: {original_sample}')
print(f'After sampling:{new_sample}')
print(f'diff: {original_sample-new_sample}')