import argparse
import enum
from importlib import import_module
import os
import multiprocessing
import numpy as np

import torch
from torch.utils.data import DataLoader
from models import *
from datasets import *
import tqdm
from tqdm.auto import tqdm

def load_model(saved_model, input_dims, device, args): # TODO : path define
    model_cls = getattr(import_module("models"), args.model)
    
    model = model_cls(
        args = args,
        input_dims = input_dims
    )

    model_path = os.path.join(saved_model, 'best.pth')

    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

@torch.no_grad()
def inference(args):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # path define
    #rating_dir = args.rating_dir
    #attr_dir = args.attr_dir
    
    # model load path
    model_dir = args.model_dir
    output_dir = args.output_dir
    # output path

    # --dataset
    print("Loading Data..")
    rating_df = pd.read_csv("/opt/ml/input/data/train/train_ratings.csv")
    rating_df["time"] = 1
    user_key = rating_df["user"].unique()
    item_key = rating_df["item"].unique()
    user_item_matrix = rating_df.pivot_table("time","user","item").fillna(0)

    dataset = InferenceDataset(user_item_matrix)
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers = 4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False
    )
    
    #n_users = dataset.get_users()# 31360 #num of users
    #n_items = dataset.get_items()# 6807 #num of items
    #n_attributes = dataset.get_attributes()
    input_dims = dataset.get_input_dim()
    
    model = load_model(model_dir, input_dims, device,args).to(device)
    model.eval()
    


    print("Calculating inference results..")
    result = list()

    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total = len(loader))
        user_cnt = 0
        for idx, batch in pbar:
            x, y = batch
            x, y = x.to(device), y.to(device) #[B, 6807]
            output = model(x) #[B]
            output = output - y
            output = output.cpu()
            for user_item in output :
                user_id = user_key[user_cnt]
                indices = np.argpartition(np.array(user_item), -10)[-10:]
                candidates = [[index, user_item[index]] for index in indices]
                candidates.sort(key=lambda x:x[1], reverse = True)
                for candidate in candidates :
                    result.append((user_id, int(item_key[candidate[0]])))
                user_cnt += 1
            #for info, score in zip(x,output):
            #    user, item = dataset.decode_offset(info.cpu())
            #    ratings[user].append([score.item(),item])
    
    #info = []
    #for user, rating in ratings.items():
    #    rating.sort(key=lambda x:x[0])
    #    for item in rating[-10:]:
    #        info.append([user,item[1]])
    
    info = pd.DataFrame(result, columns=['user','item'])
    info.to_csv(os.path.join(output_dir,f"submission.csv"),index=False)

    print("Inference Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    #Data
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size for validing (default: 1000)')
    parser.add_argument('--embedding_dim', type=int, default=512, help='embedding dimention(default: 10)')
    parser.add_argument('--out_activation', type=str, default="sigmoid")
    parser.add_argument('--hidden_activation', type=str, default="identity")
    parser.add_argument('--attr', type=str, default='genre', help='embedding dimention(default: 10)')

    parser.add_argument('--rating_dir', type=str, default='/opt/ml/input/data/train/rating.csv')
    parser.add_argument('--attr_dir', type=str, default='/opt/ml/input/data/train/genre.csv')
    

    #model parameters
    parser.add_argument('--model', type=str, default='AutoRec', help='model type (default: DeepFM)')
    parser.add_argument('--model_dir', type=str, default="./exp/experiment", help='model pth directory')
    parser.add_argument('--drop_ratio', type=float, default=0.1)

    parser.add_argument('--output_dir', type=str, default="./output", help='output directory')

    args = parser.parse_args()

    inference(args)