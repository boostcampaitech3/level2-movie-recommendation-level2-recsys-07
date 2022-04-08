import torch
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import os
import argparse
import yaml

import numpy as np

from tqdm import tqdm
from utils import fix_random_seed, increment_path, random_neg, dotdict
from dataset import SeqDataset
from model import BERT4Rec

import mlflow

EXPRIMENT_NAME = "Bert4Rec"
TRACKiNG_URI = "http://34.105.0.176:5000/"

def train(args):
    
    #-- Fix random seed
    fix_random_seed(args.seed)
    
    #-- Increment exp folder name
    save_dir = increment_path(os.path.join('./exp/', args.name))
    os.makedirs(save_dir)
    
    #-- Save current param
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(dict(args), yaml_file, default_flow_style=False)

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
        batch_size  = args.batch_size, #default batch_size = 1024 # 배치는 유저다
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
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # default: cross_entropy # mask 안한 거 빼고 함
    
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
    
    
        
    mlflow.set_tracking_uri(TRACKiNG_URI)
    mlflow.set_experiment(EXPRIMENT_NAME)
    #-- mlflow setting
    with mlflow.start_run() as run:
        mlflow.log_params(vars(args))                  # save params
        mlflow.log_artifact(f"{save_dir}/config.yaml") # config.yaml save
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
            scheduler.step()

            #-- [MLflow] Set mlflow log metrics
            mlflow.log_metrics({
                "Train/loss_average" : loss_avg.item(),
            },step = epoch)

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
                    _logits = _logits.view(-1, _logits.size(-1))   # [51200, 6808]
                    _labels = _labels.view(-1).to(device)         # 51200

                    _loss = criterion(_logits, _labels)
                    
                    correct_cnt += torch.sum((_labels == y_hat) & (_labels != 0))
                    masked_cnt += _labels.count_nonzero()
                    valid_loss += _loss
                
                valid_loss_avg = valid_loss / len(valid_loader)
                valid_acc = correct_cnt / masked_cnt
                if valid_loss_avg < best_val_loss:
                    print(f"New best model for val loss : {valid_loss_avg:.5f}! saving the best model..")
                    torch.save(model, f"{save_dir}/best.pth")
                    best_val_loss = valid_loss_avg
                    stop_counter = 0

                    #-- [MLflow] Save model artifacts to mlflow
                    mlflow.log_artifact(f"{save_dir}/best.pth")
                    
                else:
                    stop_counter += 1
                    print (f"!!! Early stop counter = {stop_counter}/{patience} !!!")
                torch.save(model, f"{save_dir}/last.pth")
                
                print(
                    f"[Val] acc : {valid_acc:4.2%}, loss: {valid_loss_avg:.5f} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:.5f}"
                )

                print (f"Epoch: {epoch + 1}, valid_acc : {valid_acc: 4.2%}, valid_loss_avg :{valid_loss_avg: .5f}")

                #-- [MLflow] mlflow valid metrics logging
                mlflow.log_metrics({
                    "Valid/accuracy" : valid_acc.item(),
                    "Valid/loss" : valid_loss_avg.item(),
                },step = epoch) 

                if stop_counter >= patience:
                    print("Early stopping")
                    break
        #-- [MLflow] save last model artifacts to mlflow
        mlflow.log_artifact(f"{save_dir}/last.pth")
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # config option
    parser.add_argument('--config', type=bool, default=True, help = 'using config using option')

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

    #-- load config.yaml
    if args.config == True:
        print("Using config.yaml option")
        with open('./config.yaml') as f: #set config.yml path
            config = yaml.safe_load(f)
        args = dotdict(config)

    print(args)

    train(args)
