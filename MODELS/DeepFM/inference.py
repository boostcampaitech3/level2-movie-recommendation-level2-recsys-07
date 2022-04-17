import argparse
from importlib import import_module
import os
import pickle
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
    # rating_dir = args.rating_dir
    # attr_dir = args.attr_dir

    #-- Inference base
    #inference_df = pd.read_csv("inference_base.csv")
    #inference_df.sort_values(by="user",axis = 0, inplace = True)

    # load data
    with open('user_dict.pickle', 'rb') as fr:
        user_dict = pickle.load(fr)

    # load data
    with open('item_dict.pickle', 'rb') as fr:
        item_dict = pickle.load(fr)
    print("users :", len(user_dict)) #31360
    print("items :", len(item_dict)) #6807

    #-- best.pth model PATH
    #model_dir = args.model_dir
    
    #-- output path
    #output_dir = args.output_dir
    
    #-- Inference DataSet & DataLoader
    inference_dataset = InferenceDataset("inference_gener_writer_director_simple.csv")
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        num_workers=1,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False
    )
    
    #-- Load model with best parameter.
    #model = load_model(model_dir, input_dims, device,args).to(device)
    model = torch.load(f"{args.model_dir}/best.pth").to(device)
    model.eval()

    #-- Start INFERENCE
    print("[DEBUG] Calculating inference results..")
    
    user_list = list()
    score_list = list()
    item_list = list()
    with torch.no_grad():
        for batch in tqdm(inference_loader):
            x = batch.to(device) # (batch_size, attr_list)
            output = model(x) #[B] ///     item[idx] = x 에 대한 확률 output[idx]
            # idx = torch.where(output >= 0.5)[0]
            
            info = x.cpu()
            #scores = output.index_select(0,idx).cpu().tolist()
            scores = output.cpu().tolist()
            users = info[:,0].tolist()
            items = info[:,1].tolist()

            user_list += users
            item_list += items
            score_list += scores            
            # preds = torch.cat((x,output.unsqueeze(1)),dim =1) # [B , 4]
            # rating = torch.cat((rating, preds.cpu()), dim = 0)
    
    np_user_list = np.array(user_list)
    np_item_list = np.array(item_list)
    np_score_list = np.array(score_list)


    #-- Select Top 10 items
    print ("[INFO] Select Top 10 Items..")

    users = list()
    items = list()
    item_offset = inference_dataset.offsets[1] # item code offest
    
    for user_code, u_id in tqdm(user_dict.items()):
        u_id = int(u_id)

        idx = np.where(np_user_list == user_code)[0].tolist()

        item_score = np_score_list.take(idx) #user code 에 해당하는 item_score
        item_ = np_item_list.take(idx) # user code에 해당하는 item
        top10_idx = np.argpartition(item_score, -10)[-10:] # 상위 10개 index 추출

        top10_item = [int(item_dict[code - item_offset]) for code in item_.take(top10_idx)] #top 10(item code -> item id)
        user_id = [u_id] * 10

        users += user_id
        items += top10_item

    result = np.vstack((users,items)).T

    info = pd.DataFrame(result, columns=['user','item'])
    info.to_csv("./output/rating_gener_writer_director_df_100_simple_inf.csv",index=False)
    print("Inference Done!")

    print("Testing recall@10...")

    # 학습에 사용된 user만 uniq_user에 저장
    uniq_user = list(user_dict.values())
    print (f"Number of users : {len(uniq_user)}")

    with open("/opt/ml/input/workspace/BERT4Rec/data/answers.json", "r") as json_file:
        answer = json.load(json_file)

    # movielens-20m과 submission을 비교하여 Recall@10 값 계산
    submission_df = info
    recall_result = []

    # 각 유저마다 recall@10 계산하여 list에 저장
    for user in tqdm(uniq_user):
        user = int(user)
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
    parser.add_argument('--batch_size', type=int, default=1024, help='input batch size for validing (default: 1000)')
    parser.add_argument('--embedding_dim', type=int, default=10, help='embedding dimention(default: 10)')
    parser.add_argument('--attr', type=str, default=["genre", "writer"], help='embedding dimention(default: 10)')

    #model parameters
    parser.add_argument('--model', type=str, default='DeepFM', help='model type (default: DeepFM)')
    parser.add_argument('--model_dir', type=str, default="./exp/rating_gener_writer_director_df_100_simple", help='model pth directory')
    parser.add_argument('--drop_ratio', type=float, default=0.1)

    parser.add_argument('--output_dir', type=str, default="./output", help='output directory')

    args = parser.parse_args()

    print (args)
    inference(args)