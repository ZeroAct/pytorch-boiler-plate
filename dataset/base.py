from abc import ABC, abstractmethod

from torch.utils.data import Dataset, DataLoader


class DatasetInterface(ABC):
    @abstractmethod
    def __len__(self):
        ...
    
    @abstractmethod
    def __getitem__(self):
        ...
        
    @abstractmethod
    def preprocess(self):
        ...
    
    @abstractmethod
    def postprocess(self):
        ...
    

class BaseDataset(Dataset):
    
    def __init__(self, opt):
        
        super(BaseDataset, self).__init__()
        
        self.opt = opt
        
        self.params = {
                'num_workers': opt.num_workers,
                'batch_size' : opt.batch_size,
                'shuffle'    : opt.shuffle,
                'drop_last'  : opt.drop_last
            }
    
    def data_loader(self):
        return DataLoader(self, **self.params)
    