from models.base import BaseModel

from torch import nn

class Classifier(BaseModel):
    
    def __init__(self, opt):
        
        super(Classifier, self).__init__(opt)
        
        self.module = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU()
            )
    
    def forward(self, x):
        x = self.module(x)
        
        return x
    
    