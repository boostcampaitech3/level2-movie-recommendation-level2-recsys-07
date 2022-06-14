import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import sparse
from dataset import *
from dataloader import *
from loss import *
from model import *
from torch.nn import Softmax
import os
import bottleneck as bn
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="MVAE", help="Choose between MVAE and DAE")
parser.add_argument("--experiment_num", type=int, default=1)
parser.add_argument("--recall_test", type=bool, default=True, help = "종테기 실행 여부")
args = parser.parse_args()


if torch.cuda.is_available():
        cuda_aval = True
device = torch.device("cuda" if cuda_aval else "cpu")
print("Now using device : ", device)
###################

print("Loading Data...")
# Load Data
uim = pd.read_csv("/opt/ml/input/workspace/CF/Non-DL/FISM/user_movie_interaction.csv")
item_id = np.array(list(uim)[1:]) # save item_ids 6807개
user_id = uim["user"].to_numpy() # save user_ids 31360개
uim_np = uim.to_numpy()[:, 1:] # 첫 번째 열은 user_id 이기 때문에 제거하고 상호작용 행렬 numpy 만듦
print("Done!")


#load model
f = open("model.pt", 'rb')
model  = torch.load(f).to(device)


#inference
print("Inference start...")
model.eval()
result = list()
for i in range(31360) : 
    user = user_id[i] # 유저 아이디
    data_np = uim_np[i] # 유저 시청 행렬 추출
    data = torch.FloatTensor(data_np) 
    data = data.reshape(1,-1).to(device) # Multi VAE 모델에서 필요한 차원으로 변경 [6807] -> [1, 6807]
       
    if args.model == "MVAE" :
        probability, _, _ = model(data) # 만약 모델이 Multi VAE라면 output이 3개
    else :
        probability = model(data) # 만약 모델이 DAE인 경우 output이 아이템별 확률값만

    probability_np = np.array(probability.cpu().detach())[0]
    probability_np[data_np.nonzero()] = -np.inf #이미 시청한 영화는 제거

    idx = np.argpartition(probability_np, -10)[-10:]
    items = item_id[idx]
    for item in items :
        result.append((user, item))
    if i%5000 == 0 :
        print(f"{i}/31360 complete")

print("Done!")


# Making Submission File
print("Making Submission File...")
info = pd.DataFrame(result, columns=['user','item'])
info.to_csv("submission.csv",index=False)
print("Inference Done!")