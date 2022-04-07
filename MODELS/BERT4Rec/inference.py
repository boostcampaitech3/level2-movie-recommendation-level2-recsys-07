import numpy as np
import pandas as pd
import argparse
import torch
import os

from torch.utils.data import DataLoader
from model import BERT4Rec
from tqdm import tqdm

def load_model(saved_model, num_user, num_item, device, args):
    model = BERT4Rec(num_user, num_item, args.hidden_units, args.num_heads, args.num_layer, args.max_seq_len, args.dropout_rate, device).to(device)
    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

@torch.no_grad()
def inference(args):

    #-- Set CUDA option
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    #-- best.pth model PATH
    model_dir = args.model_dir
    
    #-- output path
    output_dir = args.output_dir
    
    #-- Inference DataSet & DataLoader
    inference_dataset = InferenceDataset(args)
    inference_loader = DataLoader(
        inference_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        pin_memory  = use_cuda,
        drop_last   = False
    )
    
    n_users = inference_dataset.num_users # 31360
    n_items = inference_dataset.num_items # 6807 
    print (f"n_users={n_users}, n_items={n_items}")
    
    #-- Load model with best parameter.
    model = load_model(model_dir, n_users, n_items, device,args).to(device)
    model.eval()

    #-- Start INFERENCE
    print("[DEBUG] Calculating inference results..")
    rating = torch.tensor(()).cpu() # empty tensor
    with torch.no_grad():
        for batch in tqdm(inference_loader):
            x = batch.to(device) # (batch_size, attr_list)
            output = model(x) #[B] ///     item[idx] = x 에 대한 확률 output[idx]
            preds = torch.cat((x,output.unsqueeze(1)),dim =1) # [B , 4]
            rating = torch.cat((rating, preds.cpu()), dim = 0)
    
    outputs = rating.numpy()

    info = []
    #-- Select Top 10 items
    print ("[INFO] Select Top 10 Items..")
    for user_id in tqdm(range(n_users)):
        idx = np.where(outputs[:,0].astype(int) == user_id)
        user_rating = outputs[idx[0]]
        output_best10_idx = np.argpartition(user_rating[:,-1], -10)[-10:]
        output_best10 = user_rating[output_best10_idx,1]
        
        user, movie_list =  inference_dataset.decode_offset(user_id, output_best10)
        for item in movie_list:
            info.append([user,item])
        
    info = pd.DataFrame(info, columns=['user','item'])
    info.to_csv(os.path.join(output_dir, "submission.csv"),index=False)

    print("Inference Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    #Data
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for validing (default: 1000)')
    parser.add_argument('--embedding_dim', type=int, default=10, help='embedding dimention(default: 10)')
    parser.add_argument('--attr', type=str, default=["genre", "writer"], help='embedding dimention(default: 10)')

    parser.add_argument('--rating_dir', type=str, default='./data/train/rating.csv')
    parser.add_argument('--attr_dir', type=str, default='./data/train/genre_writer.csv')
    
    #model parameters
    parser.add_argument('--model_dir', type=str, default="./exp/experiment", help='model pth directory')
    parser.add_argument('--drop_ratio', type=float, default=0.1)

    parser.add_argument('--output_dir', type=str, default="./output", help='output directory')

    args = parser.parse_args()

    print (args)
    inference(args)