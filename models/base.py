import os

from abc import ABC, abstractmethod

import torch
from torch import nn


class ModelInterface(ABC):
    @abstractmethod
    def init_network(self):
        ...
    
    @abstractmethod
    def init_weights(self):
        ...
    
    @abstractmethod
    def init_optimizer(self):
        ...
    
    @abstractmethod
    def register_loss(self):
        ...
    
    @abstractmethod
    def forward(self):
        ...
        
    @abstractmethod
    def train_step(self):
        ...
        
    @abstractmethod
    def forward_with_loss(self):
        ...
    
    

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
        epoch = self.last_epoch() + 1
        
        try:
            torch.save(self.state_dict(), os.path.join(self.weights_dir, f"{epoch}_{self.name}.pth"))
        except:
            raise TypeError
    
    def load_weights(self, weights_file=None):
        
        if weights_file is None:
            weights_file = self.last_weights_file()
            
        if weights_file is None:
            print(f"There is no weights file in '{self.weights_dir}'")
        else:
            weights_file = os.path.join(self.weights_dir, weights_file)
            print(f"Loading weights from {weights_file}... ", end="")
            state_dict = torch.load(weights_file)
            self.load_state_dict(state_dict)
            print("Done!")
        
        return self.last_epoch() + 1
        
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
            return -1
    
    @staticmethod
    def get_weight_initializer(activation):
        activiation_name = activation.lower()
        
        if activiation_name in ['relu', 'leaky_relu']:  # He
            def init_weights(m):
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity=activiation_name)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    
        elif activiation_name in ['sigmoid', 'tanh']:  # xavier
            def init_weights(m):
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(activiation_name))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
                    
        elif activiation_name in ['linear']:
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                        
        elif 'conv' in activiation_name:
            def init_weights(m):
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                    nn.init.normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        else:
            raise NotImplementedError
        
        return init_weights
        
    @staticmethod
    def get_epoch(weights_file):
        return int(os.path.split(weights_file)[-1].split('_')[0])