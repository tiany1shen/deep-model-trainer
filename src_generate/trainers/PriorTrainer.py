from .Trainer import Trainer
import models

import torch
from easydict import EasyDict
from torch.nn import functional as F
from torchvision.utils import make_grid
from einops import rearrange

from utils.options import save_config

class GptTrainer(Trainer):
    def __init__(self, config: EasyDict):
        super().__init__(config)
        self._prepare_encoder()
    
    def _prepare_encoder(self):
        self.encoder = getattr(models, self.config.encoder.name)(self.config.encoder.params)
        with open(self.config.encoder.checkpoint_path, "rb") as f:
            state_dict = torch.load(f)['model']
        self.encoder.load_state_dict(state_dict)
        self.encoder = self.accelerator.prepare(self.encoder)
        self.encoder.eval()
        
    def _compute_loss(self, inputs, targets):
        _, onehot_code = self.encoder.encode(inputs)
        indices = rearrange(onehot_code, 'b d h w -> b (h w) d').argmax(dim=-1)
        labels = targets
        # indices: (batch_size, sen_len)
        # labels: (batch_size, )
        
        cls_logit, word_logits = self.model(indices, labels)
        
        cls_loss = F.cross_entropy(cls_logit, labels)
        word_logits = rearrange(word_logits, 'b hw v -> b v hw')
        predict_loss = F.cross_entropy(word_logits, indices)
        
        return {"Class_Loss": cls_loss, "Predict_Loss": predict_loss}, dict(zip(self.loss_names, self.loss_weights))
    
    @torch.no_grad()
    def _eval_epoch(self, epoch):
        return
        self.model.eval()
        sen, label = self.unwrap_model.generate()
        b = sen.shape[0]
        onehot_code = F.one_hot(sen, self.encoder.codebook_num)
        onehot_code = rearrange(onehot_code, 'b n d -> (b n) d')
        quant_emb = self.encoder.quantizer.dequantize(onehot_code)
        quant_emb = rearrange(quant_emb, '(b h w) c -> b c h w', h=4, w=4)
        imgs = self.encoder.decoder(quant_emb)
        
        grid = rearrange(imgs, 'b c h w -> () c h (b w)')
        
        for tracker in self.accelerator.trackers:
            tracker.log_images({"samples": grid}, epoch)
            tracker.log({"samples": str(label.item())}, epoch)
    
    @torch.no_grad()
    def eval(self):
        # self.model.eval()
        # sen, label = self.unwrap_model.generate()
        # b = sen.shape[0]
        # onehot_code = F.one_hot(sen, self.encoder.codebook_num)
        # onehot_code = rearrange(onehot_code, 'b n d -> (b n) d')
        # quant_emb = self.encoder.quantizer.dequantize(onehot_code)
        # quant_emb = rearrange(quant_emb, '(b h w) c -> b c h w', h=4, w=4)
        # imgs = self.encoder.decoder(quant_emb)
        
        # grid = rearrange(imgs, 'b c h w -> () c h (b w)')
        
        # for tracker in self.accelerator.trackers:
        #     tracker.log_images({"samples": grid}, epoch)
        #     tracker.log({"samples": str(label.item())}, epoch)
        
        freq = torch.zeros(self.encoder.codebook_num, 7, 7)
        
        for batch_idx, (inputs, targets) in enumerate(self.eval_dataloader):
            self.print_progress(1, 1, batch_idx, len(self.eval_dataloader), 30)
            quant_emb, onehot_code = self.encoder.encode(inputs)
            freq += onehot_code.sum(dim=0).to(freq.device)
        
        freq = rearrange(freq, 'n h w -> (h w) n')
        freq = freq / freq.sum(dim=-1, keepdim=True)
        
        for tracker in self.accelerator.trackers:
            for i in range(freq.shape[0]):
                prob = freq[i].float().numpy()
                tracker.writer.add_histogram(f"freq", prob, i)
    