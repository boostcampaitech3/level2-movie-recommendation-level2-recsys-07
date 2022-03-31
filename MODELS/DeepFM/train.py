import argparse
import glob
from importlib import import_module
import multiprocessing
import os
import random
import re
import csv
import pandas as pd

import tqdm
from tqdm.auto import tqdm

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, Dataset

import mlflow

from datasets import *
from models import *
from loss import create_criterion

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

# increment_path
def increment_path(path):
    ''' Auto increment path, runs/exp0 -> runs/exp1 ...

    '''
    path = Path(path)
    if(not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

# mlflow setting
def mlflow_set():
    return

#train
def train(args):
    # seed fix    
    seed_setting(args.seed)
    
    # path increment
    save_dir = increment_path(os.path.join(r'./exp', args.name))

    # cuda setting
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # --dataset
    data_dir = 'data/train/' # TODO: args.data_dir 추가 
    rating_df = pd.read_csv(data_dir+'rating.csv')
    attr_df = pd.read_csv(data_dir+'genre.csv')  # TODO: args.attr 추가 (str)
    dataset_module = getattr(import_module("datasets"), args.dataset)
    dataset = dataset_module(args, rating_df, attr_df) # TODO

    # train_loader, valid_loader
    valid_size = int( len(dataset) * args.val_ratio) # default val_ratio = 0.2
    train_size = len(dataset) - valid_size
    
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset,[train_size,valid_size])
    
    train_loader = DataLoader(train_dataset,
        batch_size = args.batchsize, #default batch_size = 1024
        shuffle = True,
        num_workers = multiprocessing.cpu_count()//2,
        pin_memory=use_cuda,
        drop_last=True,
        # TODO : sampler
        )
    
    valid_loader = DataLoader(valid_dataset, 
        batch_size = args.batchsize, #default batch_size = 1024
        shuffle = True,
        num_workers = multiprocessing.cpu_count()//2,
        pin_memory=use_cuda,
        drop_last=True,
        # TODO : samplers
        )

    # --model
    # for input dimention setting 
    n_users = dataset.get_users()
    n_items = dataset.get_items()
    n_attributes = dataset.get_attributes()

    input_dims = [n_users,n_items,n_attributes]
    emb_dim = args.embedding_dim # default 10

    model_module = getattr(import_module("models"), args.model)
    model = model_module( #TODO : DeepFM Setting -> Generalized Setting
        args = args,
        input_dims = input_dims, 
        embedding_dim = emb_dim,
        mlp_dims = [30,20,10],
    ).to(device)

    # --loss
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    
    # --optimizer
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    # --scheduler
    scheduler_module = getattr(import_module("torch.optim.lr_scheduler"), args.scheduler)
    scheduler = scheduler_module( #TODO : StepLR Setting -> Generalized Setting
        optimizer = optimizer,
        step_size = args.lr_decay_step,
    )

    # --Earlystopping
    patience = args.early_stopping
    stop_counter = 0
    best_val_acc = 0
    best_val_loss = np.inf

    #Start Train
    for epoch in range(args.epochs):
        model.train()
        loss_value = 0
        matches = 0
        pbar = tqdm(enumerate(train_loader), total = len(train_loader))

        # train loop
        for idx, train_batch in pbar:
            x, y = train_batch
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            result = torch.round(output)
            loss = criterion(output, y.float())

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (result == y).sum().float()
            # defrag cached memory
            torch.cuda.empty_cache()

            # TODO : log interver
            if(idx + 1) % 100 == 0:
                train_loss = loss_value / 100
                train_acc = matches / args.batch_size / 100
                current_lr = optimizer.get_last_lr()
                pbar.set_postfix(
                    {
                        "Epoch" : f"[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)})",
                        "loss" : f"{train_loss:4.4}",
                        "accuracy" : f"{train_acc}",
                        "lr" : f"{current_lr}"
                    }
                )
                loss_value = 0
                matches = 0

        scheduler.step()

        # valid loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss = 0
            val_matches = 0

            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)

                output = model(x)
                result = torch.round(output)
                loss = criterion(output, y.float())
                
                val_loss += loss.item()
                val_matches += (result == y).sum().float()

            val_acc = val_matches/valid_size * 100
            val_loss = val_loss/valid_size
            best_val_loss = min(best_val_loss, val_loss)
            
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
                stop_counter = 0
            else:
                stop_counter += 1
                print(f"!!! Early stop counter = {stop_counter}/{patience}")
            torch.save(model.module.state.dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )

            if stop_counter >= patience:
                print("Early stopping")
                break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # config option
    parser.add_argument('--config', type=bool, default=False, help = 'config using option')

    # Data and model checkpoints
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--batch_size', type=int, default=1024, help='number of batch size in each eposh (default: 1024)')
    parser.add_argument('--dataset', type=str, default='RatingDataset', help='dataset type (default: dataset)')
    parser.add_argument('--model', type=str, default='DeepFM', help='model type (default: DeepFM)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--scheduler', type=str, default='StepLR', help='scheduler type (default: StepLR)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='lr decay step (default: 50)')
    parser.add_argument('--early_stopping', type=int, default=10, help='early stopping type (default: StepLR)')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate (default: 1e-2)')
    parser.add_argument('--drop_ratio', type=float, default=0.1, help='ratio for drop out (default: 0.1)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='bce_loss', help='criterion type (default: cross_entropy)')
    parser.add_argument('--embedding_dim', type=int, default=10, help='embedding dimention(default: 10)')
    parser.add_argument('--name', default='experiment1', help='model save at ./exp/{name}')
    
    # Container env
    
    args = parser.parse_args()

    if args.config == True:
        print("using config.json option")
        # TODO: 
        
    print(args)
    # Start train
    train(args)
