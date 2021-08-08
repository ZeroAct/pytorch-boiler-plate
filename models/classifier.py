from models.base import BaseModel, ModelInterface

from torch import nn
from torch import optim

class Classifier(BaseModel, ModelInterface):
    
    def __init__(self, opt):
        
        super(Classifier, self).__init__(opt)
        
        self.init_network()
        self.init_weights()
        self.init_optimizer()
        self.register_loss()
    
    
    def init_network(self):
        self.conv = nn.Sequential(
                nn.Conv2d(self.opt.in_channel, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU())
        
        self.linear = nn.Sequential(
                nn.Linear(32, self.opt.class_num)
            )
        
        
    def init_weights(self):
        self.conv.apply(self.get_weight_initializer('relu'))
        self.linear.apply(self.get_weight_initializer('linear'))
    
    
    def init_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), self.opt.lr, eval(self.opt.betas))
    
    
    def register_loss(self):
        loss_fn = nn.CrossEntropyLoss()
        def c(yhat, target):
            return loss_fn(yhat, target)
        
        self.loss_fn = c
    
    
    def forward(self, x):
        x = self.conv(x)
        x = x.mean(axis=(2, 3))
        x = self.linear(x)
        
        return x
        
    
    def train_step(self, x, target):
        loss = self.forward_with_loss(x, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.detach().cpu().numpy()
        
        
    def forward_with_loss(self, x, target):
        yhat = self.forward(x)
        return self.loss_fn(yhat, target)
    