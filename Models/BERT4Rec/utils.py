import torch
import pandas as pd
import numpy as np
import random
import glob
import re

from pathlib import Path


def fix_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    

def increment_path(path):
    path = Path(path)
    if(not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"
    
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def random_neg(l, r, s):
    # log에 존재하는 아이템과 겹치지 않도록 sampling
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def make_new_user_movie_interaction():
    print("new_user_movie_interaction.csv 만들기 시작")
    raw_data = pd.read_csv("/opt/ml/input/data/train/train_ratings.csv", header = 0)
    raw_data["time"] = 1
    unique_sid = pd.unique(raw_data['item'])
    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))

    for i in range(len(raw_data["item"])) :
        raw_data["item"].iloc[i] = show2id[raw_data["item"].iloc[i]]
        if i%100000 == 0 :
            print(f"{i} complete")
    
    user_movie_df = raw_data.pivot_table("time","user","item").fillna(0)
    user_movie_df.to_csv("./data/new_user_movie_interaction.csv")