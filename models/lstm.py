import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMNetwork(nn.Module):

    def __init__(self, input_size=768, hidden_size=16, time_size=118, num_classes=1): 

        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, bidirectional=True)

        linear_hidden_size = time_size * hidden_size * 2
         
        self.classification = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(linear_hidden_size,2048),
                    nn.Dropout(p=0.2),
                    nn.GELU(),
                    nn.Linear(2048, num_classes),
                    nn.Sigmoid()
                )

    def forward(self, x):                
        hidden = torch.zeros(2, x.size(0), self.hidden_size, requires_grad=True).cuda()
        cell = torch.zeros(2, x.size(0), self.hidden_size, requires_grad=True).cuda()
        
        x = x.transpose(0, 1)
        x, (_, _) = self.lstm(x, (hidden, cell))
        x = x.transpose(0, 1)
        
        output = self.classification(x) 

        return output

