from .encoder import Encoder, Decoder 
from easydict import EasyDict
from torch import nn, Tensor

class AutoEncoder(nn.Module):
    def __init__(self, args: EasyDict):
        super().__init__()
        self.encoder = Encoder(
            input_dim = args.encoder.input_dim,
            hidden_dims = args.encoder.hidden_dims,
            padding = args.encoder.padding
        )
        self.decoder = Decoder(
            output_dim = args.decoder.output_dim,
            hidden_dims= args.decoder.hidden_dims,
            cropping= args.decoder.cropping
        )
    
    def forward(self, x: Tensor):
        return self.decoder(self.encoder(x))