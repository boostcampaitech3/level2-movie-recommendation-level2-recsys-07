import torch
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import os
import argparse

import numpy as np

from tqdm import tqdm
from utils import fix_random_seed, increment_path, random_neg
from dataset import SeqDataset
from model import BERT4Rec

def train(args):
    
    #-- Fix random seed
    fix_random_seed(args.seed)
    
    #-- Increment exp folder name
    save_dir = increment_path(os.path.join('./exp/', args.name))
    os.makedirs(save_dir)
    
    #-- Use CUDA if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #-- DataSet
    train_dataset = None
    valid_dataset = None
    if args.data_split == "split_by_user":
        dataset = SeqDataset(args) # train_input[:-1]
        valid_size = int(len(dataset) * args.val_ratio) # default val_ratio = 0.2
        train_size = len(dataset) - valid_size
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset,[train_size,valid_size])
    elif args.data_split == "leave_one_out":
        train_dataset = SeqDataset(args, option="leave_one_out") # train_input[:-2]
        valid_dataset = SeqDataset(args, option="split_by_user") # train_input[:-1]
        dataset = train_dataset #added to get num_item, num_user
        
    print (f"[DEBUG] DataSet has been loaded")

    #-- DataLoader: train_loader, valid_loader
    train_loader = DataLoader(train_dataset,
        batch_size  = args.batch_size, #default batch_size = 1024
        shuffle     = True,
        pin_memory  = use_cuda
    )
    valid_loader = DataLoader(valid_dataset, 
        batch_size  = args.batch_size, #default batch_size = 1024
        shuffle     = True,
        pin_memory  = use_cuda
    )
    print (f"[DEBUG] DataLoader has been loaded")

    #-- model
    num_user = dataset.num_user
    num_item = dataset.num_item
    model = BERT4Rec(num_user, num_item, args.hidden_units, args.num_heads, args.num_layer, args.max_seq_len, args.dropout_rate, device).to(device)
    
    #-- loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # default: cross_entropy
    
    #-- optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    #-- scheduler
    scheduler = CosineAnnealingLR(
        optimizer = optimizer,
        eta_min   = args.eta_min,
        T_max     = args.tmax
    )

    #-- Earlystopping
    patience        = args.patience
    stop_counter    = 0
    best_val_acc    = 0
    best_val_loss   = np.inf
    
    #-- Start Train
    print (f"[DEBUG] Start of TRAINING")
    
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0
        
        tqdm_bar = tqdm(train_loader)
        
        for idx, (log_seqs, labels) in enumerate(tqdm_bar):
            logits = model(log_seqs)
            
            # size matching
            logits = logits.view(-1, logits.size(-1))   # [51200, 6808]
            labels = labels.view(-1).to(device)         # 51200
            
            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss_sum += loss
            loss.backward()
            optimizer.step()
            
            tqdm_bar.set_description(f'Epoch: {epoch + 1:3d}| Step: {idx:3d}| Train loss: {loss:.5f}')
        
        loss_avg = loss_sum / len(train_loader)
        # print (f"Epoch: {epoch}, loss average: {loss_avg: .5f}")
        scheduler.step()
        #-- validataion
        torch.cuda.empty_cache()
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            masked_cnt = 0
            correct_cnt = 0

            for _log_seqs, _labels in valid_loader:

                _logits = model(_log_seqs)

                y_hat = _logits[:,:].argsort()[:,:,-1].view(-1)

                # size matching
                _logits = _logits.view(-1, _logits.size(-1))   # [6400, 6808]
                _labels = _labels.view(-1).to(device)         # 6400

                _loss = criterion(_logits, _labels)
                
                correct_cnt += torch.sum((_labels == y_hat) & (_labels != 0))
                masked_cnt += _labels.count_nonzero()
                valid_loss += _loss
            
            valid_loss_avg = valid_loss / len(valid_loader)
            valid_acc = correct_cnt / masked_cnt
            print (f"Epoch: {epoch}, valid_acc : {valid_acc: 4.2%}, valid_loss_avg :{valid_loss_avg: .5f}")
        

    #-- validation
    # model.eval()
    # NDCG    = 0.0 # NDCG@10
    # HIT     = 0.0 # HIT@10
    # RECALL  = 0.0 # Recall@10

    # num_item_sample = 100
    # num_user_sample = 1000 # validation with 1000 users
    # users = np.random.randint(0, num_user, num_user_sample)
    
    # user_train = dataset.user_train
    # user_valid = dataset.user_valid
    
    # for u in tqdm(users):
    #     seq = (user_train[u] + [num_item + 1])[-args.max_seq_len:]
    #     user_seen = set(user_train[u] + user_valid[u])
    #     item_idx = np.array([user_valid[u][0]] + [random_neg(1, num_item + 1, user_seen) for _ in range(num_item_sample)])
        
    #     with torch.no_grad():
    #         predictions = - model(np.array([seq]))      # [batch_size x tokens x (num_item + 1)]
    #         predictions = predictions[0][-1][item_idx]  # sampling
            
    #         # top10_items = predictions.argsort()[:10]
    #         # top10_items = item_idx[top10_items.cpu().numpy()]
            
    #         rank = predictions.argsort().argsort()[0].item() # 0번째 아이템은 상위 몇번째?
        
    #     if rank < 10: # @10
    #         NDCG += 1 / np.log2(rank + 2)
    #         HIT += 1
            
    # print(f'NDCG@10: {NDCG / num_user_sample}| HIT@10: {HIT / num_user_sample}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    #-- DataSet Arguments
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--train_rating_path', type=str, default="./data/train_ratings.csv")
    parser.add_argument('--data_split', type=str, default="split_by_user")

    
    #-- DataLoader Arguments
    parser.add_argument('--batch_size', type=int, default=1024, help='number of batch size in each epoch (default: 1024)')
    
    #-- Trainer Arguments
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr_decay_step', type=int, default=30, help='lr decay step (default: 20)')
    parser.add_argument('--patience', type=int, default=10, help='early stopping type (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--eta_min', type=float, default=1e-4)
    parser.add_argument('--tmax', type=int, default=200)
    
    
    #-- Model Arguments
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='ratio for drop out (default: 0.1)')
    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--mask_prob', type=float, default=0.15)
    parser.add_argument('--hidden_units', type=int, default=50)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_layer', type=int, default=2)
    
    #-- Experiment Arguments
    parser.add_argument('--name', type=str, default='experiment', help='model save at ./exp/{name}')
    
    
    args = parser.parse_args()

    print(args)
    train(args)
