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
    
    @torch.no_grad()
    def eval(self):
        self.model.eval()
        
        for step in range(10):
            self.print_progress(1, 1, step, 10)
            labels = torch.arange(10).long().cuda()
            indices = torch.zeros(10, 49).long().cuda()
            for i in range(49):
                # greedy search
                _, word_logits = self.model(indices, labels)
                prob = F.softmax(word_logits[:, i], dim=-1)
                indices[:, i] = prob.multinomial(1).squeeze()
            indices = rearrange(indices, 'b (h w) -> b () h w', h=7, w=7)
            
            imgs = self.encoder.decode(indices)
            imgs = self.eval_dataloader.dataset.inv_transforms()(imgs)
            grid = rearrange(imgs, 'b c h w -> () c h (b w)')
            
            import time
            
            for tracker in self.accelerator.trackers:
                tracker.log_images({"sample": grid}, step=step)
                time.sleep(1)
