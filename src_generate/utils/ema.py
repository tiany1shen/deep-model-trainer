import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def unwrap_ddp_state_dict(state_dict):
    new_state_dict = {}
    for name, param in state_dict.items():
        new_state_dict[name[7:]] = param
    return new_state_dict
        
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model 
        self.decay = decay 
        self.shadow = {}
        self.backup = {}
        
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"{name} not in ema.shadow"
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    # before evaluating or checkpointing, apply shadow weights to the model
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    # after evaluating or checkpointing, restore the model weights
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        
    def save(self, path):
        is_ddp = isinstance(self.model, DDP)
        if is_ddp:
            torch.save(unwrap_ddp_state_dict(self.shadow), path)
        else:
            torch.save(self.shadow, path)
        print(f"EMA saved to {path}")
        
    def load_state_dict(self, state_dict):
        is_ddp = isinstance(self.model, DDP)
        if is_ddp:
            wrap_name = lambda name: "module." + name
        else:
            wrap_name = lambda name: name
        for name, param in state_dict.items():
            self.shadow[wrap_name(name)] = param
        self.check_compatiable()
    
    def check_compatiable(self):
        for name, param in self.shadow.items():
            if name not in self.model.state_dict():
                raise ValueError(f"Parameter {name} not in model")
            if param.shape != self.model.state_dict()[name].shape:
                raise ValueError(f"Parameter {name} shape mismatch")
