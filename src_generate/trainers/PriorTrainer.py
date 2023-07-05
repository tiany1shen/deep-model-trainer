from .Trainer import Trainer
import models

import torch
from easydict import EasyDict
from torch.nn import functional as F
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
        
    def _register_custom_metrics(self):
        self.loss_names = ["label_loss", "predict_loss"]
        self.metric_names = []
        self.tracker.register(self.loss_names + self.metric_names)
        
    def _compute_loss(self, inputs, targets):
        onehot_code = self.encoder.encode(inputs)
        indices = onehot_code.argmax(dim=1)
        sentences = rearrange(indices, 'b h w -> b (h w)')
        label = rearrange(targets, 'b -> b ()')
        
        cls_logit, word_logits = self.model(sentences, label)
        
        cls_logit = rearrange(cls_logit, 'b () c -> b c')
        label = rearrange(label, 'b () -> b')
        cls_loss = F.cross_entropy(cls_logit, label)
        
        word_logits = rearrange(word_logits, 'b d c -> b c d')
        predict_loss = F.cross_entropy(word_logits, sentences)
        
        return {"label_loss": cls_loss, "predict_loss": predict_loss}, {"label_loss": 1.0, "predict_loss": 1.0}
    
    def _log_metrics(self, epoch):
        pass 
    
    @torch.no_grad()
    def _eval_epoch(self, epoch):
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
    def eval(self, epoch=0):
        if not self.debug:
            run = f"trial-{self.config.trial_index}"
            self.accelerator.init_trackers(run)
            save_config(self.config, self.trial_dir / 'config.yaml')
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