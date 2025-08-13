
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class PAD20(Dataset):
    def __init__(self, metadata_path, image_folder=None, split='train', features=None):
        super().__init__()
        self.metadata = pd.read_csv(metadata_path, index_col=0)
        self.metadata = self.metadata[self.metadata['split'] == split]
        all_features = ['itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation', 
                         'age_group', 'diameter', 'region', 'ACK', 'BCC', 'NEV', 'MEL', 'SEK', 'SCC']

        self.diagnostic_number_to_label = {n:l for n, l in zip(self.metadata['diagnostic_number'], self.metadata['diagnostic'])}

        if features:
            for ft in features:
                if ft not in all_features:
                    raise ValueError(f'Invalid feature: {ft}')
            self.features = features
        else:
            self.features = all_features

    def to_label(self, diagnostic_number):
        return self.diagnostic_number_to_label[diagnostic_number]

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        return row.name, \
                torch.tensor(row.loc[self.features].to_numpy(dtype=np.float32)), \
                torch.tensor(row.loc['diagnostic_number'], dtype=torch.long),