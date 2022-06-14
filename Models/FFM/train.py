
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pickle

from model import *

#seed fix
def seed_setting(seed):

    # cpu seed
    torch.manual_seed(seed)

    # GPU seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you use multi GPU

    # CuDDN option
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # numpy rand seed
    np.random.seed(seed)

    # random seed
    random.seed(seed)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    tqdm_bar = tqdm(dataloader)

    for batch, (X, y) in enumerate(tqdm_bar):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            tqdm_bar.set_description(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

    train_loss /= num_batches
    
    return train_loss


def test_loop(dataloader, model, loss_fn, task):
    num_batches = len(dataloader)
    test_loss, y_all, pred_all = 0, list(), list()

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() / num_batches
            y_all.append(y)
            pred_all.append(pred)
    
    y_all = torch.cat(y_all).cpu()
    pred_all = torch.cat(pred_all).cpu()
    
    if task == 'reg':
        err = abs(pred_all - y_all).type(torch.float).mean().item()
        print(f"Test Error: \n  MAE: {(err):>8f} \n  Avg loss: {test_loss:>8f}")
    else:
        err = roc_auc_score(y_all, torch.sigmoid(pred_all)).item()
        print(f"Test Error: \n  AUC: {err:>8f} \n  Avg loss: {test_loss:>8f}")
    
    return err, test_loss

def train_and_test(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs, task):
    train_loss, test_err, test_loss = list(), list(), list()
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss.append(train_loop(train_dataloader, model, loss_fn, optimizer))
        test_result = test_loop(test_dataloader, model, loss_fn, task)
        test_err.append(test_result[0])
        test_loss.append(test_result[1])
        print("-------------------------------\n")
        # model save
        torch.save(model, f"FFM3_{t+1}epoch.pth")
    print("Done!")
    
    return train_loss, test_err, test_loss

###############################################################################
# 주어진 결과와 정확히 비교하기 위한 random seed 고정
###############################################################################

seed = 42  # 바꾸지 마시오!
seed_setting(seed)

ffm_df = pd.read_csv('rating_gener_writer_df_100.csv')

col_len = list() #[31360, 6807, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2027]

for col in ffm_df.columns:
    if col == "rating":
        continue
    category = ffm_df[f"{col}"].astype('category')
    col_len.append(len(ffm_df[f"{col}"].astype('category').cat.categories))
    ffm_df[f"{col}"] = category.cat.codes


train_X, test_X, train_y, test_y = train_test_split(
    ffm_df.loc[:, ffm_df.columns != 'rating'], ffm_df['rating'], test_size=0.2, random_state=seed)
print('학습 데이터 크기:', train_X.shape, train_y.shape)
print('테스트 데이터 크기:', test_X.shape, test_y.shape)
######## Dataset Load ##########

# PyTorch의 DataLoader에서 사용할 수 있도록 변환 
train_dataset_ffm = TensorDataset(torch.LongTensor(np.array(train_X)), torch.Tensor(np.array(train_y)))
test_dataset_ffm = TensorDataset(torch.LongTensor(np.array(test_X)), torch.Tensor(np.array(test_y)))

######## Hyperparameter ########

batch_size = 64
data_shuffle = True
task = 'clf'
factorization_dim = 8
epochs = 10
learning_rate = 0.001
gpu_idx = 0

################################

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# cuda setting
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_dataloader_ffm = DataLoader(train_dataset_ffm,num_workers = 4, batch_size=batch_size, shuffle=data_shuffle)
test_dataloader_ffm = DataLoader(test_dataset_ffm,num_workers = 4, batch_size=batch_size, shuffle=data_shuffle)

field_dims = col_len # 각 col의 길이
model = FieldAwareFM(field_dims, factorization_dim, device=device).to(device)

loss_fn = nn.MSELoss().to(device) if (task == 'clf') else nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001, amsgrad=True)

train_loss, test_err, test_loss = train_and_test(train_dataloader_ffm, test_dataloader_ffm, 
                                                 model, loss_fn, optimizer, epochs, task)

