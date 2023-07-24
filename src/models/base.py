import torch
from torch import nn

def format_number(x: int, raw_size: bool = False):
    if raw_size:
        return str(x)
    if x > 1e9:
        return f"{x / 1e9:.2f} B"
    if x > 1e6:
        return f"{x / 1e6:.2f} M"
    if x > 1e3:
        return f"{x / 1e3:.2f} K"
    return str(x)

class BaseModel(nn.Module):
    @torch.no_grad()
    def show_pipeline(self, *input_shape):
        x = torch.randn(*input_shape)
        pipeline = "\nPileline:"
        for name, child in self.named_children():
            x = child(x)
            pipeline += "\n" + name.ljust(16) + str(list(x.shape))
        
        print("Input Shape:".ljust(16), list(input_shape))
        print("Output Shape:".ljust(16), list(x.shape))
        print(pipeline)
        
    @torch.no_grad()
    def count_parameters(self, raw_size=False):
        parameter_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buffer_num = sum(b.numel() for b in self.buffers())
        if parameter_num != 0:
            print(f"Total Parameters: {format_number(parameter_num, raw_size)}")
        if buffer_num != 0:
            print(f"Total Buffers:\t  {format_number(buffer_num, raw_size)}")
    
    @torch.no_grad()
    def summary(self, *input_shape):
        print(f"Model: {self.__class__.__name__}")
        print("========================================")
        self.count_parameters(raw_size=False)
        print("========================================")
        self.show_pipeline(*input_shape)
        print("========================================")
