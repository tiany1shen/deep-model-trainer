from torch import nn, Tensor
from torch.nn import functional as F
 
class ResidualLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self._conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self._bn_1 = nn.BatchNorm2d(out_channels)
        self._conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self._bn_2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self._identity = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self._identity = nn.Identity()
            
    def forward(self, x: Tensor):
        identity = self._identity(x)
        x = F.relu(self._bn_1(self._conv_1(x)))
        x = self._bn_2(self._conv_2(x))
        return F.relu(identity + x)
    
class DownSampleLayer(ResidualLayer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__(in_channels, out_channels, kernel_size)
        self._conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=kernel_size//2)
        self._identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

class UpSampleLayer(ResidualLayer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__(in_channels, out_channels, kernel_size)
        self._conv_1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size+1, stride=2, padding=kernel_size//2)
        self._identity = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        
class Encoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]):
        super().__init__()
        in_channels = input_dim
        self.input_layer = nn.Identity()
        self.layers = nn.ModuleList([])
        for out_channels in hidden_dims:
            modules = []
            modules.extend([
                DownSampleLayer(in_channels, out_channels, kernel_size=3),
                ResidualLayer(out_channels, out_channels, kernel_size=3),
                ResidualLayer(out_channels, out_channels, kernel_size=3)
            ])
            self.layers.append(nn.Sequential(*modules))
            in_channels = out_channels
        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels, output_dim, kernel_size=1),
            nn.BatchNorm2d(output_dim),
        )
    
    def forward(self, x: Tensor):
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)
    
class Decoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]):
        super().__init__()
        in_channels = input_dim
        self.input_layer = nn.Identity()
        self.layers = nn.ModuleList([])
        for out_channels in hidden_dims:
            modules = []
            modules.extend([
                UpSampleLayer(in_channels, out_channels, kernel_size=3),
                ResidualLayer(out_channels, out_channels, kernel_size=3),
                ResidualLayer(out_channels, out_channels, kernel_size=3)
            ])
            self.layers.append(nn.Sequential(*modules))
            in_channels = out_channels
        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels, output_dim, kernel_size=1),
            nn.BatchNorm2d(output_dim),
        )
    def forward(self, x: Tensor):
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x).clamp(-1.0, 1.0)