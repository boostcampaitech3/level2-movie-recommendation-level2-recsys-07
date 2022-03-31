import argparse
from importlib import import_module
import os
import multiprocessing
from sklearn.utils import shuffle

import torch
from torch.utils.data import DataLoader
from models import *

import tqdm
from tqdm.auto import tqdm

def load_model(saved_model, input_dims, device, args): # TODO : path define
    model_cls = getattr(import_module("model"), args.model)
    
    model = model_cls(
        args = args,
        input_dims = input_dims,
        embedding_dim = args.embedding_dim,
        mlp_dims = [30,20,10],
    )

    model_path = os.path.join(saved_model, 'best.pth')

    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

@torch.no_grad()
def inference(args):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # path define
    # model load path
    # data load path
    # output path

    # TODO : load data from path
    dataset_module = getattr(import_module("dataset"), args.dataset)
    dataset = dataset_module(
        # TODO : 
    )
    
    # TODO :  data loader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle = False,
        num_workers = multiprocessing.cpu_count()//2,shuffle=False,
        pin_memory=use_cuda,
    )
    
    # TODO : load model
    n_users = dataset.get_users()
    n_items = dataset.get_items()
    n_attributes = dataset.get_attributes()
    input_dims = [n_users,n_items,n_attributes]
    
    model = load_model(model_dir, input_dims,device).to(device)
    model.eval()

    # TODO : how to get top 10?
    print("Calculating inference results..")
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total = len(loader))
        for idx, batch in pbar:
            output = model(batch)

    print("Inference Done!")

    #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    #Data
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size for validing (default: 1000)')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--embedding_dim', type=int, default=10, help='embedding dimention(default: 10)')

    #model parameters


    arsg = parser.parse_args()

    os.makedirs()

    inference(args)