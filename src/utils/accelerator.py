from accelerate import Accelerator, DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False, find_unused_parameters=False)

accelerator = Accelerator(
    cpu=False, 
    mixed_precision=None, 
    gradient_accumulation_steps=3,
    split_batches=True,
    log_with=None, 
    project_dir=None, 
    kwargs_handlers=[ddp_kwargs]
)
accelerator.print(accelerator.num_processes)

from torch.utils.data import TensorDataset, DataLoader
from torch import randn, nn
from torch.optim import Adam

class mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
    def forward(self, x):
        return self.mlp(x)

net = mlp()
optimizer = Adam(net.parameters(), lr=0.001)
dataset = TensorDataset(randn(size=(30, 2)))
loader = DataLoader(dataset, batch_size=4)
net, loader, optimizer = accelerator.prepare(net, loader, optimizer)


for step, batch in enumerate(loader):
    accelerator.print(batch[0])
    with accelerator.accumulate(net):
        output = net(batch[0])
        loss = output.mean()
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    accelerator.print(next(iter(net.parameters())))
    accelerator.print(next(iter(net.parameters())).grad)
        