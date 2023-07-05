import torch 
from torch import nn, Tensor
from einops import repeat
import numpy as np


class GPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_cls = args.num_class
        self.num_vocab = args.num_vocab
        self.dim = args.dim
        self.sen_len = args.sen_len
        
        self.cls_embedding = nn.Embedding(self.num_cls, self.dim)
        self.word_embedding = nn.Embedding(self.num_vocab, self.dim)
        self.position_embedding = nn.Embedding(self.sen_len+1, self.dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(args.num_layer):
            self.layers.append(
                nn.TransformerEncoderLayer(
                    d_model=self.dim,
                    nhead=args.num_head,
                    dim_feedforward= 4 * self.dim,
                    batch_first=True,
                    norm_first=True
            ))
        self.out = SperateOutput(self.dim, self.num_cls, self.num_vocab)
    
    def forward(self, sen, label):
        # sen: (batch_size, sen_len)
        # label: (batch_size, 1)
        cls_emb = self.cls_embedding(label)
        sen_emb = self.word_embedding(sen)
        
        emb = torch.cat([cls_emb, sen_emb], dim=1)
        pe = self.position_embedding(torch.arange(self.sen_len+1, device=emb.device)).unsqueeze(0)
        h = emb + pe
        
        mask = generate_mask(h)
        for layer in self.layers:
            h = layer(h, src_mask=mask)

        cls_logit, word_logits = self.out(h)
        return cls_logit, word_logits
    
    def look_up_embeddings(self, sen, label):
        cls_emb = self.cls_embedding(label)
        sen_emb = self.word_embedding(sen)
        
        emb = torch.cat([cls_emb, sen_emb], dim=1)
        return emb
    
    @torch.no_grad()
    def generate(self, label=None):
        if label is None:
            label = torch.randint(self.num_cls, (1, 1)).to(self.cls_embedding.weight.device)
        sen = torch.zeros(label.shape[0], self.sen_len).to(label.device).long()
        
        for i in range(self.sen_len):
            cls_logit, word_logits = self.forward(sen, label)
            word_code = word_logits.argmax(dim=-1)[:, i]
            sen[:, i] = word_code
        
        return sen, cls_logit.argmax(dim=-1)
            
        
class SperateOutput(nn.Module):
    def __init__(self, dim, num_cls, num_vocab) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.cls_out = nn.Linear(dim, num_cls)
        self.word_out = nn.Linear(dim, num_vocab)
    
    def forward(self, x):
        x = self.mlp(x)
        return self.cls_out(x[:, :1, :]), self.word_out(x[:, 1:, :])

def generate_mask(tensor):
    size = tensor.shape[1]
    "Mask out subsequent positions."
    attn_shape = (size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return (torch.from_numpy(subsequent_mask) == 1).to(tensor.device)