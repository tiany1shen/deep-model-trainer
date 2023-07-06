import torch

from .quantizer import VectorQuantizer_EMA_Reset
from .encoder import Encoder, Decoder
from easydict import EasyDict
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange, parse_shape


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
            output_dim=args.output_dim,
            hidden_dims=args.hidden_dims)

class VaeDecoder(Decoder):
    def __init__(self, args: EasyDict):
        super().__init__(
            input_dim=args.input_dim,
            output_dim = args.output_dim, 
            hidden_dims = args.hidden_dims)

def flatten(x: Tensor):
    if len(x.shape) == 3:
        shape_x = parse_shape(x, 'b c t')
    if len(x.shape) == 4:
        shape_x = parse_shape(x, 'b c h w')
    return rearrange(x, 'b c ... -> (b ...) c'), shape_x

def unflatten(x: Tensor, shape: tuple):
    if "c" in shape:
        del shape["c"]
    if len(shape) == 2:
        return rearrange(x, '(b t) c -> b c t', **shape)
    if len(shape) == 3:
        return rearrange(x, '(b h w) c -> b c h w', **shape)
        
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
        flat_emb, shape_emb = flatten(emb)
        quant_emb, _, commit_loss, ppl = self.quantizer(flat_emb)
        quant_emb = unflatten(quant_emb, shape_emb)
        recon_x = self.decoder(quant_emb)
        return recon_x, commit_loss, ppl
    
    @torch.no_grad()
    def encode(self, x: Tensor):
        emb = self.encoder(x)
        flat_emb, shape_emb = flatten(emb)
        quant_emb, onehot_code = self.quantizer.quantize(flat_emb)
        quant_emb = unflatten(quant_emb, shape_emb)
        onehot_code = unflatten(onehot_code, shape_emb)
        return quant_emb, onehot_code
    
    def code2emb(self, onehot_code: Tensor):
        flatten_code, shape_code = flatten(onehot_code)
        quant_emb = torch.matmul(flatten_code, self.quantizer.codebook)
        quant_emb = unflatten(quant_emb, shape_code)
        return quant_emb
    
    def index2emb(self, word: Tensor):
        flatten_word, shape_word = flatten(word)
        flatten_code = F.one_hot(flatten_word.squeeze(-1), self.codebook_num).float()
        quant_emb = torch.matmul(flatten_code, self.quantizer.codebook)
        quant_emb = unflatten(quant_emb, shape_word)
        return quant_emb

    @torch.no_grad()
    def decode(self, onehot_code: Tensor):
        """ 
        Args: 
          onehot_code: (b, c, ...) tensor. Channel shape can be: 
            c = 1: when `onehot_code.dtype == torch.int64`, the input is actually indices of codewords
                   otherwise means latent embeddings are 1-dim or codebook size is only 1, which are rare cases.
            c = codebook_num: this means the input is one-hot codes
            c = latent_dim: this means the input is quantized embeddings
        """
        # when `onehot_code` is actually indices of codewords
        if onehot_code.dtype == torch.int64:
            quant_emb = self.index2emb(onehot_code)
        # when `onehot_code` is one-hot code
        elif onehot_code.shape[1] == self.codebook_num:
            quant_emb = self.code2emb(onehot_code)
        # when `onehot_code` is actually quantized embedding
        elif onehot_code.shape[1] == self.latent_dim:
            quant_emb = onehot_code
        return self.decoder(quant_emb)
        