from .Trainer import Trainer

import torch
from easydict import EasyDict
from torch.nn import functional as F
from einops import rearrange

from utils.options import save_config

class VqVaeTrainer(Trainer):
    def __init__(self, config: EasyDict):
        super().__init__(config)
        
    def _register_custom_metrics(self):
        self.loss_names = ["recon_loss", "commit_loss"]
        self.metric_names = ["perplexity_metric"]
        self.tracker.register(self.loss_names + self.metric_names)
        
    def _compute_loss(self, inputs, targets):
        recon, commitment_loss, perplexity = self.model(inputs)
        recon_loss = F.mse_loss(recon, inputs, reduction='mean')
        commit_loss = commitment_loss
        return dict(
            recon_loss=recon_loss,
            commit_loss=commit_loss,
            # perplexity_metric=perplexity,
        ), dict(
            recon_loss=1.0,
            commit_loss=1.0,
        )
    
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
    
    def _log_metrics(self, epoch):
        pass
    
    @torch.no_grad()
    def eval(self, epoch=0):
        if not self.debug:
            run = f"trial-{self.config.trial_index}"
            self.accelerator.init_trackers(run)
            save_config(self.config, self.trial_dir / 'config.yaml')
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
            
            
            