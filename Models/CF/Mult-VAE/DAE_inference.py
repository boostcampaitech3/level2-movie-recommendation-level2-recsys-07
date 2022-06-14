import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import sparse
from dataset import *
from dataloader import *
from loss import *
from model import *


if __name__ == '__main__':
    ## 각종 파라미터 세팅
    parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')

    parser.add_argument('--data', type=str, default='/opt/ml/input/data/train/',
                        help='Movielens dataset location')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.00,
                        help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
    parser.add_argument('--total_anneal_steps', type=int, default=200000,
                        help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='largest annealing parameter')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    args = parser.parse_args([])

    # Set the random seed manually for reproductibility.
    torch.manual_seed(args.seed)

    #만약 GPU가 사용가능한 환경이라면 GPU를 사용
    if torch.cuda.is_available():
        args.cuda = True

    device = torch.device("cuda" if args.cuda else "cpu")
    print("Now using device : ", device)

    print(args)

    ###################
    print("Load and Preprocess Movielens dataset")
    # Load Data
    DATA_DIR = args.data
    raw_data = pd.read_csv(os.path.join(DATA_DIR, 'train_ratings.csv'), header=0)

    print("원본 데이터\n", raw_data)

    raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_uc=5, min_sc=0)
    #제공된 훈련데이터의 유저는 모두 5개 이상의 리뷰가 있습니다.
    print("5번 이상의 리뷰가 있는 유저들로만 구성된 데이터\n",raw_data)
    #user_activity, item_popularity = get_count(raw_data, 'user'), get_count(raw_data, 'item')
    print("유저별 리뷰수\n",user_activity)
    print("아이템별 리뷰수\n",item_popularity)

    # User Indices (not shuffled)
    unique_uid = user_activity.index

    n_users = unique_uid.size #31360
    n_heldout_users = 3000

    ###############################################################################
    # Data set
    ###############################################################################

    # Split Train/Validation/Test User Indices
    tr_users = unique_uid[:]


    ##훈련 데이터에 해당하는 아이템들
    #Train에는 전체 데이터를 사용합니다.
    train_plays = raw_data.loc[raw_data['user'].isin(tr_users)]

    ##아이템 ID
    unique_sid = pd.unique(train_plays['item']) #6807
    unique_sid.sort() #순서 정렬

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    pro_dir = os.path.join(DATA_DIR, 'pro_sg')

    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    #Validation과 Test에는 input으로 사용될 tr 데이터와 정답을 확인하기 위한 te 데이터로 분리되었습니다.

    train_data = numerize(train_plays, profile2id, show2id)
    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)


    print("Done!")

    ###############################################################################
    # Load data
    ###############################################################################

    loader = DataLoader(args.data)

    n_items = loader.load_n_items()
    train_data = loader.load_data('train')

    N = train_data.shape[0]
    idxlist = list(range(N))

    ###############################################################################
    # Load the model
    ###############################################################################

    p_dims = [200, 600, n_items]
    f = open(args.save, 'rb')
    model  = torch.load(f)
    ###############################################################################
    # Training code
    ###############################################################################

    best_n100 = -np.inf
    update_count = 0

    model.eval()
    result = list()
    user_cnt = 0
    for batch_idx, start_idx in enumerate(range(0, N, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, N)
        data = train_data[idxlist[start_idx:end_idx]]
        data_tensor = naive_sparse2tensor(data).to(device)
        
        
        recon_batch = model(data_tensor)
        recon_batch = recon_batch.cpu().detach()
        recon_batch[data.nonzero()] = -np.inf
        batch_users = recon_batch.shape[0]
        for row in recon_batch :
            user_id = unique_uid[user_cnt]
            idx = np.argpartition(row, -10)[-10:]
            item_id = unique_sid[idx]
            for item in item_id :
                result.append((user_id, item))
            user_cnt += 1

    info = pd.DataFrame(result, columns=['user','item'])
    info.to_csv("submission.csv",index=False)

    print("Inference Done!")