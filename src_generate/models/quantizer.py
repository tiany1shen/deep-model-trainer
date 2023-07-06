import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class VectorQuantizer_EMA_Reset(nn.Module):
    """ 
    VQ-VAE layer: Input any tensor to be quantized. Use EMA to update embeddings.
    
    Args:
        codebook_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        codebook_num (int): the number of vectors in the quantized space.
        decay (float): decay for the moving averages.
        reset_threshold (float): threshold of resetting the codebook.
        epsilon (float): small float constant to avoid numerical instability.
    
    Attributes:
        dim (int): the dimensionality of the tensors in the quantized space.
        num_cb (int): the number of vectors in the quantized space.
        decay (float): decay for the moving averages.
        reset_prob (float): probability of resetting the codebook. If one cluster 
          has not been updated for a long time, it will be reset.
        epsilon (float): small float constant to avoid numerical instability.
        
    Methods:
        quantize(x): quantize input tensor x to onehot code.
        
        dequantize(onehot_code): dequantize onehot code to continuous vector in 
          the quantized space.
          
        forward(x): quantize input tensor x, and return the following tuple:
          (quantized tensor, onehot code, commitment loss, perplexity)
          - quantized tensor: [N, D]
          - onehot code: [N, num_cb]
          - commitment loss: scalar tensor to metric the loss of input tensor x
          - perplexity: scalar tensor to metric the information of quantized tensor 
              perplexity = exp(-sum(p * log(p))),   p: distribution of onehot code
    """
    def __init__(self, codebook_dim: int, codebook_num: int, 
                 decay: float = 0.99, reset_threshold: float = 0.01, epsilon: float = 1e-7):
        super().__init__()
        self.dim = codebook_dim
        self.num_cb = codebook_num
        self.decay = decay
        self.reset_prob = reset_threshold / codebook_num
        self.epsilon = epsilon
        
        # initialize embeddings as buffers
        self._init_buffers()
    
    
    def _init_buffers(self):
        embeddings = torch.zeros(self.num_cb, self.dim)
        self.register_buffer("codebook", embeddings)
        
        cluster_prob = torch.ones(self.num_cb) / self.num_cb
        self.register_buffer("cluster_prob", cluster_prob)
        

    def _tile(self, x):
        # x: [N, D]
        num_emb_x, dim_emb = x.shape
        if num_emb_x < self.num_cb:
            n_repeats = (self.num_cb + num_emb_x - 1) // num_emb_x
            std = 0.01 / torch.sqrt(dim_emb).to(x.device)
            out = repeat(x, 'n d -> (n r) d', r=n_repeats)
            out = out + torch.randn_like(out).to(x.device) * std
        else:
            out = x
        return out[:self.num_cb]
    
    
    def quantize(self, x):
        # x: [N, D]
        # Return: onehot code [N, num_cb], dtype: float32
        distance = (
            torch.sum(x**2, dim=-1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=-1) - \
            2 * torch.matmul(x, self.codebook.t())
        )
        _, code_idx = torch.min(distance, dim=-1)
        
        onehot_code = F.one_hot(code_idx, self.num_cb).float()
        x_quantized = torch.matmul(onehot_code, self.codebook)
        return x_quantized, onehot_code
    
    @torch.no_grad()
    def _compute_perplexity(self, onehot_code):
        cluster_count = torch.sum(onehot_code, dim=0)
        prob = cluster_count / torch.sum(cluster_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + self.epsilon)))
        return perplexity
    
    @torch.no_grad()
    def _update_codebook(self, x, onehot_code):
        # x: [N, D]
        # onehot_code: [N, num_cb]
        
        embbeding_sum = torch.matmul(onehot_code.t(), x)  # [num_cb, D]
        cluster_count = torch.sum(onehot_code, dim=0)  # [num_cb]
        new_emb_centers = embbeding_sum / (rearrange(cluster_count, 'n -> n ()') + self.epsilon)
        
        self.codebook = self.codebook * self.decay + new_emb_centers * (1 - self.decay)
        self.cluster_prob = self.cluster_prob * self.decay + cluster_count / cluster_count.sum() * (1 - self.decay)
        
        # 1 for centers to be reset, 0 for centers to be kept
        reset_mask = (self.cluster_prob < self.reset_prob).float()
        self._reset_codebook(x, reset_mask)
        
        # batch perplexity
        prob = cluster_count / cluster_count.sum()
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + self.epsilon)))
        return perplexity
    
    def _reset_codebook(self, x, reset_mask=None):
        if reset_mask is None:
            reset_mask = torch.ones(self.num_cb).to(x.device) # [num_cb, ]
            
        num_reset = torch.sum(reset_mask).item()
        # reset codebook to random values in `x`
        remained_codebook = (1 - rearrange(reset_mask, 'n -> n ()')) * self.codebook
        new_codebook = rearrange(reset_mask, 'n -> n ()') * self._tile(x)
        self.codebook = remained_codebook + new_codebook
        
        # reset cluster prob an normalize it to 1-sum 
        remained_prob = (1 - reset_mask) * self.cluster_prob 
        new_prob = reset_mask * torch.ones_like(self.cluster_prob).to(x.device) / self.num_cb    # sum: num_reset / num_cb
        
        self.cluster_prob = new_prob + remained_prob / (remained_prob.sum() + self.epsilon) * (1 - num_reset / self.num_cb)
 
    
    def forward(self, flat_x):
        # flat_x: [N, D]
        
        if self.training and self.codebook.norm() == 0:
            self._reset_codebook(flat_x)
        
        x_quantized, onehot_code = self.quantize(flat_x)
        
        if not self.training:
            return x_quantized, onehot_code, None, self._compute_perplexity(onehot_code)
            
        # update codebook
        perplexity = self._update_codebook(flat_x, onehot_code)
        
        commitment_loss = F.mse_loss(x_quantized.detach(), flat_x)
        x_quantized = flat_x + (x_quantized - flat_x).detach()
        
        return x_quantized, onehot_code, commitment_loss, perplexity
 