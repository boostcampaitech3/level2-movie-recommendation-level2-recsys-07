import argparse
import enum
from importlib import import_module
import os
import multiprocessing
from sklearn.utils import shuffle
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
    rating_dir = args.rating_dir
    attr_dir = args.attr_dir
    
    # model load path
    model_dir = args.model_dir
    
    # output path
    output_dir = args.output_dir
    
    dataset = InferenceDataset(args, rating_dir, attr_dir)
    
    loader = DataLoader(
        dataset,
        batch_size = args.batch_size,
        num_workers = 4,
        shuffle = False,
        pin_memory = use_cuda,
        drop_last = False
    )
    
    n_users = dataset.get_users() # 31360 #num of users
    n_items = dataset.get_items() # 6807 #num of items
    n_attributes = dataset.get_attributes()
    input_dims = [n_users, n_items, n_attributes]
    
    model = load_model(model_dir, input_dims, device,args).to(device)
    model.eval()


    print("Calculating inference results..")
    #ratings = { value:[] for key, value in dataset.user_dict.items()}
    rating = torch.tensor(()).cpu() # empty tensor
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total = len(loader))
        for idx, batch in pbar:
            x = batch.to(device) #[B, 3]
            output = model(x) #[B] ///     item[idx] = x 에 대한 확률 output[idx]
            
            preds = torch.concat((x,output.unsqueeze(1)),dim =1) # [B , 4]

            rating = torch.concat((rating, preds.cpu()), dim = 0)
    
    outputs = rating.numpy()

    info = []
    for user_id in range(n_users):
        idx = np.where(outputs[:,0].astype(int) == user_id)
        user_rating = outputs[idx[0]]
        output_best10_idx = np.argpartition(user_rating[:,-1], -10)[-10:]
        output_best10 = user_rating[output_best10_idx,1]
        
        user, movie_list =  dataset.decode_offset(user_id, output_best10)
        for item in movie_list:
            info.append([user,item])
        
    info = pd.DataFrame(info, columns=['user','item'])
    info.to_csv(os.path.join(output_dir,f"submission.csv"),index=False)

    print("Inference Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    #Data
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--embedding_dim', type=int, default=10, help='embedding dimention(default: 10)')
    parser.add_argument('--attr', type=str, default='writer', help='embedding dimention(default: 10)')

    parser.add_argument('--rating_dir', type=str, default='/opt/ml/input/data/train/rating.csv')
    parser.add_argument('--attr_dir', type=str, default='/opt/ml/input/data/train/writer.csv')
    
    #model parameters
    parser.add_argument('--model', type=str, default='DeepFM', help='model type (default: DeepFM)')
    parser.add_argument('--model_dir', type=str, default="/opt/ml/input/exp/expriment", help='model pth directory')
    parser.add_argument('--drop_ratio', type=float, default=0.1)

    parser.add_argument('--output_dir', type=str, default="/opt/ml/input/output", help='output directory')

    args = parser.parse_args()

    inference(args)