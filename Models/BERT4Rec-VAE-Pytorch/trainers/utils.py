import torch
import pickle

import numpy as np
import pandas as pd

def recall(scores, labels, k):
    scores = scores
    labels = labels
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / torch.min(torch.Tensor([k]).to(hit.device), labels.sum(1).float())).mean().cpu().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    scores = scores
    labels = labels
    answer_count = labels.sum(1)

    labels_float = labels.float()
    
    # print ("--" * 50, "labels")
    # print (labels_float)
    # print (labels.shape)
    
    rank = (-scores).argsort(dim=1) # 각 batch 마다 점수가 가장 높은 아이템의 index가 맨 앞으로 온다
    cut = rank
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k] # 아이템 상위 k개 확인
       hits = labels_float.gather(1, cut) # 상위 k개의 index를 가져온다
    #    print ("--" * 50, "cut")
    #    print (cut)
    #    print (cut.shape)
       metrics['Recall@%d' % k] = \
           (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item()

       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float())
       dcg = (hits * weights.to(hits.device)).sum(1)
       idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
       ndcg = (dcg / idcg).mean()
       metrics['NDCG@%d' % k] = ndcg.cpu().item()

    return metrics

def get_best_10(scores, user_seen):
    rank = (-scores).argsort(dim=1)
    cut = rank
    
    best_10 = list()
    
    for movie in cut[0]:
        if movie not in user_seen:
            best_10.append(movie)
        if len(best_10) == 10:
            break
            
    best_10 = torch.tensor(best_10)
    best_10 = best_10.unsqueeze(0)
    
    # print (f"user's recommend list = {best_10}")
    
    return best_10

def make_inference_file(recommend_list:dict, args):
    train_ratings_df = pd.read_csv("/opt/ml/level2-movie-recommendation-level2-recsys-07/MODELS/BERT4Rec/data/train_ratings.csv")
    
    item_ids = train_ratings_df['item'].unique() # 6807
    user_ids = train_ratings_df['user'].unique() # 31360

    #-- Re-index user, item
    # CAUTION : user starts with index 1 (1 ~ 31360)
    #           item starts with index 1 (1 ~ 6807) 
    item2idx = pd.Series(data=np.arange(len(item_ids)) + 1, index=item_ids)
    user2idx = pd.Series(data=np.arange(len(user_ids)) + 1, index=user_ids)
    
    idx2item = pd.Series(dict((v, k) for k, v in item2idx.iteritems()))
    idx2user = pd.Series(dict((v, k) for k, v in user2idx.iteritems()))
    
    submission_list = list()
    
    # 유저랑 아이템 index 복원시키기
    for user in recommend_list.keys():
        rec_items = recommend_list[user].squeeze()
        
        if user % 100 == 1:
            print (f"user {user}'s recommend list = {rec_items}")
        
        # print ("rec_items : ", rec_items)
        decoded_user = idx2user[user]
        decoded_item = []
        for ri in rec_items:
            decoded_item = idx2item[ri.item()]
            submission_list.append((decoded_user, decoded_item))
            
    submission_df = pd.DataFrame(submission_list, columns=['user','item'])
    submission_df.to_csv("/opt/ml/submission_BERT4Rec.csv", index=False)
    print("Inference Done!")
        
        
    
    # csv로 만들어서 저장하기
    
    