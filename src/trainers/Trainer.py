import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import models as Models
import datasets as Datasets
from utils import Optimizer, MetricTracker, SyncMetricTracker
from utils.options import save_config
from utils.progress import print_progress

import torch
from torch.utils.data import DataLoader
from easydict import EasyDict
from pathlib import Path
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as DDPKwargs


class Trainer:
    def __init__(self, config: EasyDict):
        self.config = config
        self.debug = self.config.debug_epoch or self.config.debug_iter
        torch.backends.cudnn.benchmark = True
        self._build()
         
    def _build(self):
        self._dir_setting()
        self._ddp_setting()
        self._build_models()
        self._build_dataloaders()
        self._build_optimizer()
        self._build_trackers()
        
    def _ddp_setting(self):
        cpu = getattr(self.config, self.config.mode).use_cpu
        gradient_accumulation_steps = self.config.gradient_accumulation_steps
        if self.config.log_metrics:
            log_with = 'tensorboard'
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
        self.checkpoint_path = getattr(self.config, self.config.mode).checkpoint_path
        if self.checkpoint_path is not None and Path(self.checkpoint_path).exists():
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.accelerator.print(f"Loaded model from {self.checkpoint_path}")
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
            optimizer = getattr(Optimizer, self.config.train.optimizer)(
                    params=self.model.parameters(), lr=self.config.train.learning_rate
                )
            if self.checkpoint_path is not None and Path(self.checkpoint_path).exists():
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                optimizer.load_state_dict(checkpoint['optimizer'])
                optimizer.param_groups[0]['capturable'] = True
            else:
                optimizer.zero_grad()
            
            self.optimizer = self.accelerator.prepare(optimizer)
            
    def _dir_setting(self):
        self.experiment_dir = Path('OUTPUTS')
        self.experiment_name = self.config.experiment_name
        self.trial_index = self.config.trial_index
        
        self.log_dir = Path(self.experiment_dir, self.experiment_name, 'logs')
        
        trial_name = f"{self.config.mode}-{self.config.model.name}-{self.trial_index}"
        self.trial_dir = Path(self.experiment_dir, self.experiment_name, trial_name)
        if self.config.mode == 'train':
            self.checkpoint_dir = Path(self.trial_dir, 'checkpoints')
            
        if self.debug:
            return 
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self, 'checkpoint_dir'):
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
       
    def _build_trackers(self):
        # build accelerator logger
        if not self.debug:
            run = f"{self.config.mode}-{self.config.model.name}-{self.config.trial_index}"
            self.accelerator.init_trackers(run)
            save_config(self.config, self.trial_dir / 'config.yaml')
            
        # build metric trackers
        self.tracker = MetricTracker() if self.accelerator.num_processes == 1 else SyncMetricTracker()
        self._register_custom_metrics()
        
    def train(self):
        assert self.config.mode == 'train', f"Trainer should be in train mode, got {self.config.mode} mode."
        train_config = self.config.train
        self.model.train()
        
        self.tracker.register('Total_Loss')
        
        total_step = (self.start_epoch - 1) * int(len(self.train_dataloader) // self.config.gradient_accumulation_steps)
        # Training Loop
        for epoch in range(self.start_epoch, self.start_epoch + train_config.num_epochs):
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.model.train()
                
                self.print_progress(epoch, self.start_epoch + train_config.num_epochs-1, batch_idx, len(self.train_dataloader))
                with self.accelerator.accumulate(self.model):
                    loss_dict, weight_dict = self._compute_loss(batch)
                    
                    self.tracker.update({name: loss.detach().float() for name, loss in loss_dict.items()})
                    # print(loss_dict, weight_dict)
                    # exit()
                    total_loss = sum(loss_dict[name] * weight_dict[name] for name in self.loss_names)
                    self.tracker.update({'Total_Loss': total_loss.detach().float()})
                    
                    self.accelerator.backward(total_loss)
                    
                    self.optimizer.step()
                    # self.scheduler.step()
                    self.optimizer.zero_grad()
                
                iter_completed = (batch_idx + 1) % self.config.gradient_accumulation_steps == 0
                if iter_completed:
                    total_step += 1
                    # if self.ema is not None:
                    #     self.ema.update(self.model)
                    self._log_after_step(total_step)

                    if self.config.debug_iter:
                        self.accelerator.print('Debug mode, only run 1 step.')
                        return
            
            if train_config.need_eval:
                if epoch % train_config.eval_interval == 0:
                    self._eval_epoch(epoch)
                self._log_after_epoch(epoch)
                
            if epoch % train_config.save_interval == 0:
                self._save_checkpoint(epoch)

            if self.config.debug_epoch:
                self.accelerator.print('Debug mode, only run 1 epoch.')
                return
    
    def _register_custom_metrics(self):
        self.loss_names = [] 
        self.metric_step_names = []
        self.metric_epoch_names = []
        self.accelerator.print(f" Tracking Scalars during {self.config.mode}:")
        if hasattr(self.config, "train"):
            if self.config.train.loss.names is not None:
                self.loss_names += [name + "_Loss" for name in self.config.train.loss.names]
                self.loss_weights = self.config.train.loss.weights
            if self.config.train.metric_names is not None:
                self.metric_step_names += [name + "_Metric" for name in self.config.train.metric_names]
            self.accelerator.print(f"\tper ITER:  {self.loss_names + ['Total_Loss'] + self.metric_step_names}")              
        if hasattr(self.config, "eval"):
            if self.config.eval.metric_names is not None:
                self.metric_epoch_names += [name + "_Metric" for name in self.config.eval.metric_names]
            self.accelerator.print(f"\tper EPOCH: {self.metric_epoch_names}")
        self.tracker.register(self.loss_names + self.metric_step_names + self.metric_epoch_names)
    
    def _log_scalars(self, names, epoch, reduction="last"):
        if self.debug:
            return
        metrics = self.tracker.fetch(names, reductions=reduction)
        self.tracker.register(names)
        self.accelerator.log(metrics, step=epoch)
        
    def _log_after_step(self, step):
        if step % self.config.train.log_interval == 0:
            self._log_scalars(self.loss_names + ['Total_Loss'], step, reduction="mean")
            self._log_scalars(self.metric_step_names, step, reduction="last")
            # self._log_img_hist(step=step, **kwargs)
        
    def _log_after_epoch(self, epoch):
        if epoch % self.config.train.eval_interval == 0:
            self._log_scalars(self.metric_epoch_names, epoch, reduction="last")
        # self._log_img_hist(step=epoch, **kwargs)
    
    def _compute_loss(self, batch):
        """
        Compute loss for a batch of data.
        
        DDP model wrapped by Accelerator will not inherit costumized methods. Use `self.unwrap_model` to access these methods.
        
        Returns:
          loss_dict: dict of losses
          weight_dict: dict of weights for each loss
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def _eval_epoch(self, epoch):
        raise NotImplementedError
    
    @torch.no_grad()
    def eval(self):
        raise NotImplementedError
        
        
    def _save_checkpoint(self, epoch):
        if self.debug:
            return
        self.accelerator.wait_for_everyone()
        self.model.eval()
        if self.accelerator.is_main_process:
            checkpoint = {"epoch": epoch}
            checkpoint["model"] = self.unwrap_model.state_dict()
            checkpoint["optimizer"] = self.optimizer.state_dict()
            checkpoint_path = Path(self.checkpoint_dir, f"epoch-{epoch}.pth")
            torch.save(checkpoint, checkpoint_path)
    
    @property
    def unwrap_model(self):
        if self.ddp_wrapped:
            return self.model.module
        else:
            return self.model
        
    def print_progress(self, epoch, num_epoch, batch_idx, epoch_len, length=10):
        if self.debug:
            return
        progress_bar = print_progress(epoch, num_epoch, batch_idx+1, epoch_len, length=length)
        end_char = "\n" if epoch == num_epoch and batch_idx == epoch_len - 1 else "\r"
        self.accelerator.print(progress_bar, end=end_char)
        
    @property
    def after_step_hooks(self):
        return [
            lambda self: self._log_after_step(),
            
        ]
    
    @property
    def after_epoch_hooks(self):
        return [
            lambda self: self._log_after_epoch(),
        ]
        
    def _apply_hooks(self, hooks):
        for hook in hooks:
            hook(self)
            
     