from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.transforms.functional import crop
 
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
        
class CenterCrop2d(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.height = size[0]
        self.width = size[1]
        
    def forward(self, x: Tensor):
        _, _, h, w = x.shape
        top = h // 2 - self.height // 2
        left = w // 2 - self.width // 2
        return crop(x, top, left, self.height, self.width)
    
class Clamp(nn.Module):
    def __init__(self, *values) -> None:
        super().__init__()
        if values is None:
            self.lower = -1.0
            self.upper = 1.0
        elif len(values) == 1:
            self.lower = -values[0]
            self.upper = values[0]
        elif len(values) == 2:
            self.lower = values[0]
            self.upper = values[1]
        else:
            raise NotImplementedError()
    
    def forward(self, x):
        return x.clamp(self.lower, self.upper)
        
class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], padding: int):
        super().__init__()
        in_channels = input_dim
        self.input_layer = nn.ZeroPad2d(padding)
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
    
    def forward(self, x: Tensor):
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, output_dim: int, hidden_dims: list[int], cropping: int):
        super().__init__()
        in_channels = hidden_dims[0]
        hidden_dims = hidden_dims[1:] + [output_dim]
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
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            Clamp(1.0),
            CenterCrop2d(cropping)
        )
    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)