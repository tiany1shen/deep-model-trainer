from .Trainer import Trainer

import torch
from easydict import EasyDict
from torch.nn import functional as F
from einops import rearrange

class VqVaeTrainer(Trainer):
    def __init__(self, config: EasyDict):
        super().__init__(config)
        
    def _register_custom_metrics(self):
        self.loss_names = ["recon_loss", "commit_loss"]
        self.metric_names = ["perplexity_metric"]
        self.tracker.register(self.loss_names + self.metric_names)
        
    def _compute_loss(self, inputs, targets):
        recon, commitment_loss, perplexity = self.unwrap_model(inputs)
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
        pass
    
    def _log_metrics(self, epoch):
        pass
    
    @torch.no_grad()
    def eval(self):
        run = f"trial-{self.config.trial_index}"
        self.accelerator.init_trackers(run)
        
        self.model.eval()
        for (inputs, targets) in self.eval_dataloader:
            recon, _, _ = self.unwrap_model(inputs)
            pic = torch.cat([inputs, recon], dim=0)
            pic = self.eval_dataloader.dataset.inv_transforms()(pic)
            grid = rearrange(pic, "(nr nc) c h w -> () c (nr h) (nc w)", nr=2, nc=10)
            
            print(recon.max(), recon.min())
            exit()
            
            self.accelerator.trackers[0].log_images({"image": grid}, step=10)
            break
            
            
            