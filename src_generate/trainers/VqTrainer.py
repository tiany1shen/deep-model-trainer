from .Trainer import Trainer

import torch
from easydict import EasyDict
from torch.nn import functional as F
from einops import rearrange

from utils.options import save_config

class VqVaeTrainer(Trainer):
    def __init__(self, config: EasyDict):
        super().__init__(config)
        
        
    def _compute_loss(self, inputs, targets):
        recon, commitment_loss, perplexity = self.model(inputs)
        recon_loss = F.mse_loss(recon, inputs, reduction='mean')
        commit_loss = commitment_loss
        return dict(
            Recon_Loss=recon_loss,
            Commit_Loss=commit_loss,
            Perplexity_Metric=perplexity,
        ), dict(zip(self.loss_names, self.loss_weights))
    
    @torch.no_grad()
    def _eval_epoch(self, epoch):
        self.model.eval()
        n_samples = self.config.eval.num_sample
        for (inputs, targets) in self.eval_dataloader:
            inputs = inputs[:n_samples]
            recon, _, _ = self.model(inputs)
            
            pic = torch.cat([inputs, recon], dim=0)
            pic = self.eval_dataloader.dataset.inv_transforms()(pic)
            
            grid = rearrange(pic, "(nr nc) c h w -> () c (nr h) (nc w)", nr=2, nc=n_samples)
            
            for tracker in self.accelerator.trackers:
                tracker.log_images({"image": grid}, step=epoch)
            break
    
    @torch.no_grad()
    def eval(self, epoch=0):
        self.model.eval()
        n_samples = self.config.eval.num_sample
        for (inputs, targets) in self.eval_dataloader:
            inputs = inputs[:n_samples]
            
            latent = self.unwrap_model.encode(inputs)
            recon = self.unwrap_model.decode(latent)
            pic = torch.cat([inputs, recon], dim=0)
            pic = self.eval_dataloader.dataset.inv_transforms()(pic)
            
            grid = rearrange(pic, "(nr nc) c h w -> () c (nr h) (nc w)", nr=1, nc=n_samples)
            
            for tracker in self.accelerator.trackers:
                tracker.log_images({"latent_image": grid}, step=epoch)
            break
            
            
            