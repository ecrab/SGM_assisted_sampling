import math

import torch.nn as nn
import torch


class ScoreNet(nn.Module):
    def __init__(self, in_dim=2+3, out_dim=2):
        super(ScoreNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Sequential(
                nn.Linear(self.in_dim, 64), #layer 1
                nn.ReLU(True),

                nn.Linear(64, 128), #layer 2
                nn.ReLU(True),

                nn.Linear(128, 256), #layer 3
                nn.ReLU(True),
            
                nn.Linear(256, 512), #layer 4
                nn.ReLU(True),
            
                nn.Linear(512, 256), #layer 5
                nn.ReLU(True),
            
                nn.Linear(256, 128), #layer 6
                nn.ReLU(True),
            
                nn.Linear(128, 64), #layer 7
                nn.ReLU(True),
            
                nn.Linear(64, self.out_dim), #layer 10
        )
    
    

    def forward(self, x, t, y):
        y = y.view(-1, 1)
        t = torch.cat((t - 0.5, torch.cos(2*torch.tensor(math.pi)*t)),dim=1)
        t = torch.cat((t, y), dim=1)
        x = torch.cat((x,t), dim=1)
        output = self.linear(x)
        return output
    

#The below network can be used to generate 1D samples/conditional measures rather than 2D. This is useful if you know the slow direction a priori, but the above network is a better catch-all for the example problem at hand.    
    
    
class ScoreNet2(nn.Module):
    def __init__(self, in_dim=1+5, out_dim=1):
        super(ScoreNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inner_dim = 100 
        
        self.linear = nn.Sequential(
                nn.Linear(self.in_dim, self.inner_dim), #layer 1
                nn.ReLU(True),

                nn.Linear(self.inner_dim, self.inner_dim), #layer 2
                nn.ReLU(True),

                nn.Linear(self.inner_dim, self.inner_dim), #layer 3
                nn.ReLU(True),
            
                nn.Linear(self.inner_dim, self.inner_dim), #layer 4
                nn.ReLU(True),
            
                nn.Linear(self.inner_dim, self.inner_dim), #layer 5
                nn.ReLU(True),
            
                nn.Linear(self.inner_dim, self.inner_dim), #layer 6
                nn.ReLU(True),
            
                nn.Linear(self.inner_dim, self.inner_dim), #layer 7
                nn.ReLU(True),
            
                nn.Linear(self.inner_dim, self.out_dim), #layer 8
        )
        
    
    

    def forward(self, x, t, y):
        y = y.unsqueeze(-1)
        t = torch.cat((t - 0.5, torch.cos(2*torch.tensor(math.pi)*t), torch.sin(2*torch.tensor(math.pi)*t), torch.cos(4*torch.tensor(math.pi)*t)),dim=1)
        t = torch.cat((t, y), dim=1)
        x = torch.cat((x,t), dim=1)
        output = self.linear(x)
        return output
    
    