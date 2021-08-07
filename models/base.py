import os

from abc import abstractmethod

import torch
from torch import nn

class BaseModel(nn.Module):
    
    """
    This class is an Base Model class.
    
    # === attributes === #
    + name : str
        model name
        
    + weights_dir : str
        weights directory
            this directory should contain weights files named like 
            "{epoch}_whatever.pth"
            e.g. 1_G.pth, 2_G.pth, ...
    
    
    
    # === methods === #
    + save_weights(path) : 
    
    - last_weights_file() : str
    
    """
    
    def __init__(self, opt):
        
        super(BaseModel,self).__init__()
        
        self.opt = opt
        
        self.name         = opt.name
        self.weights_dir  = opt.weights_dir
    
    def save_weights(self, epoch=None):
        epoch = self.last_epoch()
        
        try:
            torch.save(self.state_dict(), os.path.join(self.weights_dir, f"{epoch}_{self.name}.pth"))
        except:
            raise TypeError
    
    def load_weights(self, weights_file=None):
        
        if weights_file is None:
            weights_file = self.last_weights_file()
            
        if weights_file is None:
            print(f"There is no weights file in '{self.weights_dir}'")
            return
        else:
            print(f"Loading weights from {weights_file}... ", end="")
            state_dict = torch.load(weights_file)
            self.load_state_dict(state_dict)
            print("Done!")
        
    def last_weights_file(self):
        weights_files = os.listdir(self.weights_dir)
        
        if weights_files:
            weights_files.sort(key=lambda x: int(os.path.split(x)[-1].split('_')[0]))
            return weights_files[-1]
        else:
            return None
    
    def last_epoch(self):
        weights_file = self.last_weights_file()
        
        if weights_file:
            return self.get_epoch(weights_file)
        else:
            return 0
    
    @staticmethod
    def get_epoch(weights_file):
        return int(os.path.split(weights_file)[-1].split('_')[0])
    
    @abstractmethod
    def forward(self):
        pass