import torch
from torch import nn 
from easydict import EasyDict 


class MLP(nn.Module):
    def __init__(self, args: EasyDict):
        super().__init__()
        self.input_dims = [args.input_dim] + args.hidden_dims
        self.output_dims = args.hidden_dims + [args.output_dim]

        layers = []
        for i in range(len(self.input_dims)):
            layers.append(nn.Linear(self.input_dims[i], self.output_dims[i]))
            if i != len(self.input_dims) - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
    def compute_loss(self, outputs, targets):
        criterion = nn.CrossEntropyLoss()
        return criterion(outputs, targets)
    
    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs.argmax(dim=-1)