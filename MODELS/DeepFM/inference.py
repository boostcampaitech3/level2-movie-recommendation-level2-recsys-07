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
import json

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

    #-- Define Path (ratings.csv, genre_writer.csv)
    rating_dir = args.rating_dir
    attr_dir = args.attr_dir
    
    #-- best.pth model PATH
    model_dir = args.model_dir
    
    #-- output path
    output_dir = args.output_dir
    
    #-- Inference DataSet & DataLoader
    inference_dataset = InferenceDataset(args, rating_dir, attr_dir)
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False
    )
    
    n_users = inference_dataset.get_users() # 31360 # of users
    n_items = inference_dataset.get_items() # 6807 # of items
    n_attributes1 = inference_dataset.get_attributes1()
    n_attributes2 = inference_dataset.get_attributes2()
    print (f"n_users={n_users}, n_items={n_items}, n_attr1={n_attributes1}, n_attr2={n_attributes2}")
    
    input_dims = [n_users, n_items, n_attributes1, n_attributes2]
    
    #-- Load model with best parameter.
    model = load_model(model_dir, input_dims, device,args).to(device)
    model.eval()

    #-- Start INFERENCE
    print("[DEBUG] Calculating inference results..")
    rating = torch.tensor(()).cpu() # empty tensor
    with torch.no_grad():
        for batch in tqdm(inference_loader):
            x = batch.to(device) # (batch_size, attr_list)
            # print ("[DEBUG] model input x-----")
            # print (x)
            # print ("--------------------------")
            output = model(x) #[B] ///     item[idx] = x 에 대한 확률 output[idx]
            # idx = torch.where(output >= 0.5)[0]
            
            # x = x.index_select(0,idx)
            # output = output.index_select(0,idx)
            
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
    info.to_csv("./submission.csv",index=False)
    print("Inference Done!")

    print("Testing recall@10...")

    # 학습에 사용된 user만 uniq_user에 저장
    train_ratings_df = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv')
    uniq_user = train_ratings_df['user'].unique()
    print (f"Number of users : {len(uniq_user)}")

    with open("/opt/ml/input/workspace/BERT4Rec/data/answers.json", "r") as json_file:
        answer = json.load(json_file)

    # movielens-20m과 submission을 비교하여 Recall@10 값 계산
    submission_df = info # TODO: submission_file path에 맞게 수정!
    recall_result = []

    # 각 유저마다 recall@10 계산하여 list에 저장
    for user in tqdm(uniq_user):
        submission_by_user = submission_df[submission_df['user'] == user]['item']

        hit = 0
        for item in submission_by_user:
            if item in answer[str(user)]:
                hit += 1

        recall_result.append(hit / 10)

    # 전체 유저의 Recall@10의 평균 출력
    print (f"Predicted submission result of Recall@10 = {np.average(recall_result)}")    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    #Data
    parser.add_argument('--batch_size', type=int, default=4096, help='input batch size for validing (default: 1000)')
    parser.add_argument('--embedding_dim', type=int, default=20, help='embedding dimention(default: 10)')
    parser.add_argument('--attr', type=str, default=["genre", "writer"], help='embedding dimention(default: 10)')

    parser.add_argument('--rating_dir', type=str, default='./data/train/rating.csv')
    parser.add_argument('--attr_dir', type=str, default='./data/train/genre_writer.csv')
    
    #model parameters
    parser.add_argument('--model', type=str, default='DeepFM', help='model type (default: DeepFM)')
    parser.add_argument('--model_dir', type=str, default="./exp/experiment", help='model pth directory')
    parser.add_argument('--drop_ratio', type=float, default=0.1)

    parser.add_argument('--output_dir', type=str, default="./output", help='output directory')

    args = parser.parse_args()

    print (args)
    inference(args)