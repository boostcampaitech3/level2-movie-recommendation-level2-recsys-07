import pandas as pd
import torch
from torch.utils.data import Dataset

class AutoRecDataset(Dataset) :
    def __init__(self, user_item_matrix) :
        super().__init__()
        self.user_item_matrix = user_item_matrix
        
    def __len__(self) :
        return len(self.user_item_matrix)

    def get_input_dim(self) :
        return len(self.user_item_matrix.iloc[0, :])

    def __getitem__(self, index) :
        
        # index번째 유저 행 추출, 첫번째 값은 유저 id이기 때문에 제외하고 반환
        user_vector = torch.tensor(self.user_item_matrix.iloc[index, :].to_numpy()).float()
        return user_vector, user_vector

class InferenceDataset(Dataset) :
    def __init__(self, user_item_matrix) :
        super().__init__()
        self.user_item_matrix = user_item_matrix
        
    def __len__(self) :
        return len(self.user_item_matrix)

    def get_input_dim(self) :
        return len(self.user_item_matrix.iloc[0, :])

    def __getitem__(self, index) :
        # index번째 유저 행 추출, 첫번째 값은 유저 id이기 때문에 제외하고 반환
        user_vector = torch.tensor(self.user_item_matrix.iloc[index, :].to_numpy()).float()
        return user_vector, user_vector