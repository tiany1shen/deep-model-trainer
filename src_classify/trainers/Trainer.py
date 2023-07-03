import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import models as Models
import datasets as Datasets
from utils import Optimizer, MetricTracker, SyncMetricTracker
from utils.options import save_config

import torch
from torch.utils.data import DataLoader
from easydict import EasyDict
from pathlib import Path
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as DDPKwargs


class Trainer:
    def __init__(self, config: EasyDict):
        self.config = config
        torch.backends.cudnn.benchmark = True
        self._build()
        
    def train(self):
        assert self.config.mode == 'train', f"Trainer should be in train mode, got {self.config.mode} mode."
        train_config = self.config.train
        self.model.train()
        
        # init ema
        # if hasattr(train_config, 'use_ema') and train_config.use_ema:
        #     self._init_ema()
        # else:
        #     self.ema = None 
        
        # init tracker
        run = f"trial-{self.config.trial_index}"
        self.accelerator.init_trackers(run)
        save_config(self.config, self.trial_dir / 'config.yaml')
        self.tracker = MetricTracker() if self.accelerator.num_processes == 1 else SyncMetricTracker()
        self.tracker.register('total_loss')
        self._register_custom_metrics()
        
        total_step = (self.start_epoch - 1) * int(len(self.train_dataloader) // self.config.gradient_accumulation_steps)
        start_step = total_step
        # Training Loop
        for epoch in range(self.start_epoch, self.start_epoch + train_config.num_epochs):
            self.model.train()
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.model.train()
                with self.accelerator.accumulate(self.model):
                    inputs, targets = batch
                    loss_dict, weight_dict = self._compute_loss(inputs, targets)
                    
                    if len(loss_dict) > 1:
                        self.tracker.update({name: loss.detach().float() for name, loss in loss_dict.items()})
                    
                    total_loss = sum(loss_dict[name] * weight_dict[name] for name in loss_dict.keys())
                    self.tracker.update({'total_loss': total_loss.detach().float()})
                    
                    self.accelerator.backward(total_loss)
                    self.optimizer.step()
                    # self.scheduler.step()
                    self.optimizer.zero_grad()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    total_step += 1
                    # if self.ema is not None:
                    #     self.ema.update(self.model)
                    if total_step % train_config.log_interval == 0:
                        self._log_loss(total_step)
            
            if 'eval' in train_config.need_other_modes:
                if epoch % train_config.eval_interval == 0:
                    self._eval_epoch(epoch)
                    self._log_metrics(epoch)
            
            if 'sample' in train_config.need_other_modes:
                if epoch % train_config.sample_interval == 0:
                    self._sample_epoch(epoch)
                
            if epoch % train_config.save_interval == 0:
                self._save_checkpoint(epoch)
    
    def _compute_loss(self, inputs, targets):
        """
        Compute loss for a batch of data.
        
        DDP model wrapped by Accelerator will not inherit costumized methods. Use `self.unwrap_model` to access these methods.
        
        Returns:
          loss_dict: dict of losses
          weight_dict: dict of weights for each loss
        
        Example:
            >>> outputs = self.model(inputs)
            >>> loss = self.unwrap_model.compute_loss(outputs, targets)
            >>> return loss
        """
        raise NotImplementedError
    
    
    def _register_custom_metrics(self):
        raise NotImplementedError
    
    def _log_metrics(self, epoch):
        raise NotImplementedError
    
    def _log_loss(self, step):
        loss_names = [name for name in self.tracker.metrics.keys() if name.endswith('loss')]
        
        losses = self.tracker.fetch(loss_names, reductions='mean')
        self.tracker.register(loss_names)
        self.accelerator.log(losses, step=step)
            
    @torch.no_grad()
    def eval(self):
        raise NotImplementedError
        
    @torch.no_grad()
    def _eval_epoch(self, epoch):
        raise NotImplementedError
    
    @torch.no_grad() 
    def sample(self):
        raise NotImplementedError
        
    @torch.no_grad()
    def _sample_epoch(self, epoch):
        raise NotImplementedError
        
    def _save_checkpoint(self, epoch):
        self.accelerator.wait_for_everyone()
        self.model.eval()
        if self.accelerator.is_main_process:
            checkpoint = {"epoch": epoch}
            checkpoint["model"] = self.unwrap_model.state_dict()
            checkpoint_path = Path(self.checkpoint_dir, f"epoch-{epoch}.pth")
            torch.save(checkpoint, checkpoint_path)
        
    def _build(self):
        self._dir_setting()
        self._ddp_setting()
        self._build_models()
        self._build_dataloaders()
        self._build_optimizer()
        
    def _ddp_setting(self):
        cpu = self.config.use_cpu
        gradient_accumulation_steps = self.config.gradient_accumulation_steps
        if self.config.log_metrics:
            log_with = 'all'
            project_dir = self.log_dir
        else:
            log_with = None
            project_dir = None
        ddp_kwargs = DDPKwargs(broadcast_buffers=False, find_unused_parameters=False)
        self.accelerator = Accelerator(
            cpu=cpu, 
            gradient_accumulation_steps=gradient_accumulation_steps,
            split_batches=True,
            log_with=log_with, 
            project_dir=project_dir, 
            kwargs_handlers=[ddp_kwargs]
        )
        self.device = self.accelerator.device
        self.accelerator.print("Accelerator prepared successfully")
        
    def _build_models(self):
        model_name = self.config.model.name
        model_hyperparameters = self.config.model.params
        model_class = getattr(Models, model_name)
        self._init_ddp_model(model_class(model_hyperparameters))
        self.accelerator.print("Model prepared successfully")
        
    def _init_ddp_model(self, model):
        if self.config.model.checkpoint_path is not None and Path(self.config.model.checkpoint_path).exists():
            checkpoint = torch.load(self.config.model.checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.accelerator.print(f"Loaded model from {self.config.model.checkpoint_path}")
        else:
            if hasattr(model, 'init_weights'):
                model.init_weights()
            self.start_epoch = 1
            self.accelerator.print("No checkpoint found, initializing model from scratch")
        self.model = self.accelerator.prepare(model)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.ddp_wrapped = True
        else:
            self.ddp_wrapped = False
    
    # def _init_ema(self):
    #     ema_config = self.config.ema
    #     self.ema = EMA(self.model, ema_config.decay)
    #     if hasattr(ema_config, 'checkpoint_path') and Path(ema_config.checkpoint_path).exists():
    #         self.ema.load_state_dict(torch.load(ema_config.checkpoint_path, map_location=self.device))
    #     else:
    #         self.ema.register()
        
    def _build_dataloaders(self):
        if hasattr(self.config.dataset, 'train'):
            train_dataset_name = self.config.dataset.train.name
            train_dataset_class = getattr(Datasets, train_dataset_name)
            train_dataset = train_dataset_class(self.config.dataset.train.params)
            
            train_batchsize = int(self.config.train.batch_size // self.config.gradient_accumulation_steps)
            self.train_dataloader = self.accelerator.prepare(DataLoader(
                train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=8, pin_memory=True
            ))
            self.accelerator.print(f"Train dataset prepared successfully, batch size {train_batchsize}")
        else:
            self.train_dataloader = None 
        if hasattr(self.config.dataset, 'eval'):
            eval_dataset_name = self.config.dataset.eval.name
            eval_dataset_class = getattr(Datasets, eval_dataset_name)
            eval_dataset = eval_dataset_class(self.config.dataset.eval.params)
            
            eval_batchsize = int(self.config.eval.batch_size // self.config.gradient_accumulation_steps)
            self.eval_dataloader = self.accelerator.prepare(DataLoader(
                eval_dataset, batch_size=eval_batchsize, shuffle=True, num_workers=8, pin_memory=True
            ))
            self.accelerator.print(f"Eval dataset prepared successfully, batch size {eval_batchsize}")
        else:
            self.eval_dataloader = None 
              
    def _build_optimizer(self):
        if self.config.mode == 'train':
            self.optimizer = self.accelerator.prepare(
                getattr(Optimizer, self.config.train.optimizer)(
                    params=self.model.parameters(), lr=self.config.train.learning_rate
                )
            )
            self.optimizer.zero_grad()
         
    def _dir_setting(self):
        self.experiment_dir = Path('experiments')
        self.experiment_name = self.config.experiment_name
        self.trial_index = self.config.trial_index
        
        self.log_dir = Path(self.experiment_dir, self.experiment_name, 'logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.trial_dir = Path(self.experiment_dir, self.experiment_name, f"trial-{self.trial_index}")
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = Path(self.trial_dir, 'checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.config, 'sample'):
            self.sample_dir = Path(self.trial_dir, 'samples')
            self.sample_dir.mkdir(parents=True, exist_ok=True)

    @property
    def unwrap_model(self):
        if self.ddp_wrapped:
            return self.model.module
        else:
            return self.model