import pandas as pd
import torch
from torch.utils.data import Dataset

class AutoRecDataset(Dataset) :
    def __init__(self, data_dir) :
        super().__init__()
        self.user_item_matrix = pd.read_csv(data_dir)
        
    def __len__(self) :
        return len(self.user_item_matrix)

    def __getitem__(self, index) :

        # index번째 유저 행 추출, 첫번째 값은 유저 id이기 때문에 제외하고 반환
        user_vector = self.user_item_matrix.loc[index, :][1:]
        return user_vector