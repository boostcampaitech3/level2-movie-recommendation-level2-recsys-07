import numpy as np
import torch

from torch.utils.data import Dataset
from preprocess import preprocess_original, preprocess2_leave_one_out

class SeqDataset(Dataset):
    def __init__(self, args, option="split_by_user"):
        
        if option == 'split_by_user': # [:-1] : train or valid
            num_user, num_item, train_rating_df, user_train, user_valid = preprocess_original(args, option)
        elif option == 'leave_one_out': # [:-2] : train
            num_user, num_item, train_rating_df, user_train, user_valid = preprocess2_leave_one_out(args, option)
        
        self.num_user        = num_user          # 31360
        self.num_item        = num_item          # 6807
        self.train_rating_df = train_rating_df
        self.user_train      = user_train        # 유저가 마지막에 본 영화 제외한 리스트
        self.user_valid      = user_valid
        self.max_len         = args.max_seq_len  # default: 50
        self.mask_prob       = args.mask_prob    # default: 0.15


    def __len__(self):
        return self.num_user                     # 31360


    def __getitem__(self, user):
        
        user_sequence = self.user_train[user]
        tokens = []
        labels = []
        
        for item in user_sequence:
            prob = np.random.random()
            
            #-- Do masking
            # MASK -> num_item + 1, negative sample, postivie sample
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.num_item + 1) # masking
                elif prob < 0.9:
                    tokens.append(np.random.randint(1, self.num_item+1)) # negative sample + positive sample
                else:
                    tokens.append(item) # positive sample  
                labels.append(item) # label of masked or random sampled item
                
            #-- No masking
            else:
                tokens.append(item) # positive sample
                labels.append(0)    # trivial
                
        # if len(seq) > max_len: slicing
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        mask_len = self.max_len - len(tokens)

        # if len(seq) < max_len: zero padding
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        
        # X: user's watch history without last movie [masked + random sampled(neg, pos)]
        # y: user's watch history without last movie
        return torch.LongTensor(tokens), torch.LongTensor(labels)
    
