import numpy as np
import pandas as pd
import pickle 
import torch
from model import BERT4Rec
import os
import argparse
from utils import *
import yaml
import json
from tqdm import tqdm

# 맨 마지막 시점에서 item 10개 추천
def last_time(key, data, result) :
    # key = user_index, data = input data(sequence), result = model output(probability)
    t = result[-1] # dim = [1, 6808]
    t[data[0]] = -np.inf # 이미 시청한 영화 제거
    top_k_idx = np.argpartition(t, -10)[-10:] # top 10 proability 계산
    rec_item_id = item_id[top_k_idx] # 영화 추출
    user = user_id[key]
    for item in rec_item_id :
        final.append((user, item))

# item 별로 확률 값을 모두 더해서 추천
def all_sum(key, data, result) :
    t = result.sum(axis = 0)
    t[data[0]] = -np.inf # 이미 시청한 영화 제거
    top_k_idx = np.argpartition(t, -10)[-10:] # top 10 proability 계산
    rec_item_id = item_id[top_k_idx] # 영화 추출
    user = user_id[key]
    for item in rec_item_id :
        final.append((user, item))

# 5번째 time마다 top 10을 뽑아서 그 중 가장 빈도수가 높은 순서대로 추천
def top_10_per_five(key, data, result) :
    top_100 = list()
    t_cnt = 0
    for t in result :
        if t_cnt % 5 == 0 :
            t[data[0]] = -np.inf
            top_k_idx = np.argpartition(t, -10)[-10:]
            for i in top_k_idx :
                top_100.append(item_id[i])
                #top_100.append(int(idx2item[int(i)]))

    item_counts = np.unique(top_100, return_counts=True)
    counts = item_counts[1]
    top_10_idx = np.argpartition(counts, -10)[-10:]
    rec_item_id = item_counts[0][top_10_idx]

    user = user_id[key]

    for item in rec_item_id :
        final.append((user, item))

func = {"last_time" : last_time, "all_sum" : all_sum, "top_10_per_five" : top_10_per_five}

# GPU 설정
if torch.cuda.is_available() :
    cuda_aval = True
device = torch.device("cuda" if cuda_aval else "cpu")

#argparse 정의
parser = argparse.ArgumentParser()
parser.add_argument('--exp_num', type=int, default=1, help="number of experiment")
parser.add_argument("--inference_type", type=str, default="last_time")
parser.add_argument("--recall_test", type=bool, default=True, help = "종테기 실행 여부")
args = parser.parse_args()

# Loading Data
print("loading data...")
if os.path.isfile('./data/new_user_movie_interaction.csv'):
    print("new_user_movie_interaction.csv 파일이 이미 존재합니다")
else:
    print("Start to make new_user_movie_interaction.csv (13minutes)")
    make_new_user_movie_interaction()
    print("new_user_movie_interaction.csv 생성 완료")

uim = pd.read_csv("./data/new_user_movie_interaction.csv")
user_id = uim["user"].to_numpy()

with open("./data/input_data.pickle", "rb") as fr : #user_seen_dict #preprocess.py에서 만들어짐
    raw_data = pickle.load(fr)

unique_sid = pd.read_csv("/opt/ml/input/data/train/pro_sg/unique_sid.txt", header=None)
show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))

item_id = unique_sid.to_numpy().reshape(6807)
item_id = np.insert(item_id, 0, 0)
print("data loading compelete!")

#Load model
EXP_N = args.exp_num
#model = torch.load(f"./exp/experiment{EXP_N}/best.pth").to(device)
model = torch.load(f"./exp/experiment{EXP_N}/best.pth").to(device)
model.eval()

with open(f'./exp/experiment{EXP_N}/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

#inference 부분
print("\nstart inference..!")
inference_function = func[args.inference_type]
max_len = config['max_seq_len']
final = list()
cnt = 0
for key in raw_data.keys() :

    length = len(raw_data[key])
    if length < max_len : 
        dif = max_len-length
        data = [0]*dif + raw_data[key][-length:]
    else :
        data = raw_data[key][-max_len:]

    data = torch.LongTensor(data).unsqueeze(dim=0)
    result = model(data)[0].detach().cpu()
    result = result[-min(length, max_len):] # 만약 시청 영화의 개수가 max_len보다 작다면, 시청 영화의 개수만큼 행 고려

    inference_function(key, data, result)
    
    cnt+=1
    if cnt%5000 == 0 :
        print(f"{cnt}/31360 complete")

print("inference complete!")

print("making submission file...")
#Making Inference file
if not os.path.isdir('./output'):
    os.makedirs('./output')

info = pd.DataFrame(final, columns=['user','item'])
info.to_csv(f"./output/submission_B4R_{EXP_N}.csv",index=False)


if args.recall_test == True :
    print("testing recall@10...")
    # 학습에 사용된 user만 uniq_user에 저장
    train_ratings_df = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv')
    uniq_user = train_ratings_df['user'].unique()
    print (f"Number of users : {len(uniq_user)}")

    with open("/opt/ml/input/workspace/BERT4Rec/data/answers.json", "r") as json_file:
        answer = json.load(json_file)

    # movielens-20m과 submission을 비교하여 Recall@10 값 계산
    submission_df = pd.read_csv(f"./output/submission_B4R_{EXP_N}.csv") # TODO: submission_file path에 맞게 수정!
    recall_result = []

    # 각 유저마다 recall@10 계산하여 list에 저장
    for user in tqdm(uniq_user):
        submission_by_user = submission_df[submission_df['user'] == user]['item']

        hit = 0
        for item in submission_by_user:
            if item in answer[str(user)]:
                hit += 1

        recall_result.append(hit / 10)

    # 전체 유저의 Recall@10의 평균 출력
    print (f"Predicted submission result of Recall@10 = {np.average(recall_result)}")


print("Inference Done!")

