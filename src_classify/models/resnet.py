import torch
from torch import nn
from torch.nn import functional as F
 
class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), 
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.identity = nn.Identity()
        
    def forward(self, x):
        identity = self.identity(x)
        residual = self.resblock(x)
        return F.relu(identity + residual)

    
class Resnet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        in_channels = hidden_dims[0]
        self.input_layer = nn.Conv2d(input_dim, in_channels, kernel_size=1)
        self.layers = nn.ModuleList([
            ResidualLayer(in_channels, in_channels, stride=1),
        ])
        for out_channels in hidden_dims[1:]:
            self.layers.extend([
                ResidualLayer(in_channels, out_channels, stride=2),
                ResidualLayer(out_channels, out_channels, stride=1)
            ])
            in_channels = out_channels
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, output_dim)
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
    
class ResnetModel(Resnet):
    def __init__(self, args):
        super().__init__(
            args.input_dim, 
            args.output_dim, 
            args.hidden_dims
        )
    
    def compute_loss(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)
    
    @torch.no_grad()
    def predict(self, inputs, is_logit=True):
        if not is_logit:
            inputs = self.forward(inputs)
        return torch.argmax(inputs, dim=1)