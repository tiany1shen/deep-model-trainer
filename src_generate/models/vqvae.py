import torch

from .quantizer import VectorQuantizer_EMA_Reset
from .encoder import Encoder, Decoder
from easydict import EasyDict
from torch import nn, Tensor
from einops import rearrange


class VectorQuantizer(VectorQuantizer_EMA_Reset):
    def __init__(self, args: EasyDict):
        super().__init__(
            codebook_dim=args.codebook_dim, 
            codebook_num=args.codebook_num, 
            decay=args.decay,
            reset_threshold=args.reset_threshold,
            epsilon=args.epsilon
            )
        self.codebook_dim = args.codebook_dim
        self.codebook_num = args.codebook_num
        
class VaeEncoder(Encoder):
    def __init__(self, args: EasyDict):
        super().__init__(
            input_dim=args.input_dim, 
            hidden_dims=args.hidden_dims, 
            padding=args.padding)

class VaeDecoder(Decoder):
    def __init__(self, args: EasyDict):
        super().__init__(
            output_dim = args.output_dim, 
            hidden_dims = args.hidden_dims, 
            cropping = args.cropping)
        
class VQVAE(nn.Module):
    def __init__(self, args: EasyDict):
        super().__init__()
        self.encoder = VaeEncoder(args.encoder)
        self.quantizer = VectorQuantizer(args.quantizer)
        self.decoder = VaeDecoder(args.decoder)
        
        self.latent_dim = self.quantizer.codebook_dim
        self.codebook_num = self.quantizer.codebook_num
        
    def forward(self, x: Tensor):
        emb = self.encoder(x)
        b, c, h, w = emb.shape
        
        flat_emb = rearrange(emb, 'b c h w -> (b h w) c')
        quant_emb, onehot_code, commit_loss, ppl = self.quantizer(flat_emb)
        
        quant_emb = rearrange(quant_emb, '(b h w) c -> b c h w', b=b, h=h, w=w)
        recon_x = self.decoder(quant_emb)
        return recon_x, commit_loss, ppl
    
    @torch.no_grad()
    def encode(self, x: Tensor):
        emb = self.encoder(x)
        b, c, h, w = emb.shape
        
        flat_emb = rearrange(emb, 'b c h w -> (b h w) c')
        onehot_code = self.quantizer.quantize(flat_emb)
        return rearrange(onehot_code, '(b h w) d -> b d h w', b=b, h=h, w=w)

        