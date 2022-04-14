import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pickle
from model import *

import json

inference_df = pd.read_csv("inference_base.csv") 

# load data
with open('user_dict.pickle', 'rb') as fr:
    user_dict = pickle.load(fr)

# load data
with open('item_dict.pickle', 'rb') as fr:
    item_dict = pickle.load(fr)
print("users :", len(user_dict)) #31360
print("items :", len(item_dict)) #6807

inference_df.sort_values(by="user",axis = 0, inplace = True)

# cuda setting
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

batch_size = 512

inference_dataset = TensorDataset(torch.LongTensor(np.array(inference_df)))
inference_dataloader = DataLoader(inference_dataset,
                                  batch_size=batch_size,
                                  pin_memory=use_cuda,
                                  drop_last=False,
                                  shuffle=False,
                                  num_workers = 4,
                                  )

print("dataset length :", len(inference_dataloader))

model = torch.load(f"./output/FFM3_2epoch.pth").to(device)
model.eval()

user_list = list()
score_list = list()
item_list = list()

with torch.no_grad():
    cnt = 0
    for batch in tqdm(inference_dataloader):
        x = batch[0].to(device) 
        output = model(x) #[B] ///x 에 대한 점수
        #idx = torch.where(output >= 1)[0] # 점수가 1 이상인 index
        
        info = x.cpu()
        #scores = output.index_select(0,idx).cpu().tolist()
        scores = output.cpu().tolist()
        users = info[:,0].tolist()
        items = info[:,1].tolist()

        user_list += users
        item_list += items
        score_list += scores

np_user_list = np.array(user_list)
np_item_list = np.array(item_list)
np_score_list = np.array(score_list)

users = list()
items = list()
for user_code, u_id in tqdm(user_dict.items()):
    u_id = int(u_id)

    idx = np.where(np_user_list == user_code)[0].tolist()
    
    item_score = np_score_list.take(idx) #user code 에 해당하는 item_score
    item_ = np_item_list.take(idx) # user code에 해당하는 item
    top10_idx = np.argpartition(item_score, -10)[-10:] # 상위 10개 index 추출

    top10_item = [int(item_dict[code]) for code in item_.take(top10_idx)] #top 10(item code -> item id)
    user_id = [u_id] * 10

    users += user_id
    items += top10_item

result = np.vstack((users,items)).T

info = pd.DataFrame(result, columns=['user','item'])
info.to_csv("./output/FFM3_submission_2epoch.csv",index=False)

print("testing recall@10...")
# 학습에 사용된 user만 uniq_user에 저장
uniq_user = list(user_dict.values())
print (f"Number of users : {len(uniq_user)}")

with open("/opt/ml/input/workspace/BERT4Rec/data/answers.json", "r") as json_file: #answer.json 경로 지정
    answer = json.load(json_file)

# movielens-20m과 submission을 비교하여 Recall@10 값 계산
submission_df = info
recall_result = []

# 각 유저마다 recall@10 계산하여 list에 저장
for user in tqdm(uniq_user):
    submission_by_user = submission_df[submission_df['user'] == user]['item']
    user = int(user)
    hit = 0
    for item in submission_by_user:
        if item in answer[str(user)]:
            hit += 1

        recall_result.append(hit / 10)

# 전체 유저의 Recall@10의 평균 출력
print (f"Predicted submission result of Recall@10 = {np.average(recall_result)}")