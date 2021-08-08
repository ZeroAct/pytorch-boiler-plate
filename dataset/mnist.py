import os

import numpy as np
import pandas as pd

from torchvision import transforms, datasets

from dataset.base import BaseDataset, DatasetInterface


class MNist(BaseDataset, DatasetInterface):
    
    def __init__(self, opt, csv_name, test=False):
        
        super(MNist, self).__init__(opt)
        
        self.data_dir = os.path.join(self.opt.data_root, self.opt.dataset)
        
        data = pd.read_csv(os.path.join(self.data_dir, csv_name), delimiter=',').to_numpy()
        
        print(f"{len(data)} data loaded from {csv_name}!")
        
        if not test:
            self.xs = data[:, 1:].reshape(-1, 28, 28).astype(np.float32)
            self.ys = data[:, 0]
        else:
            self.xs = data.reshape(-1, 28, 28).astype(np.float32)
            self.ys = None
        
        self.transform = transforms.Compose([
                                transforms.ToTensor()
                            ])
    
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        if self.ys is not None:
            return self.preprocess(self.xs[idx]), self.ys[idx]
        else:
            return self.preprocess(self.xs[idx])
    
    def preprocess(self, x):
        return self.transform(x)
    
    def postprocess(self, x):
        ...