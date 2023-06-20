import torch 
import matplotlib.pyplot as plt
from easydict import EasyDict
from trainers import Trainer


class ClassificationTrainer(Trainer):
    def __init__(self, config: EasyDict):
        super().__init__(config)
    
    def _compute_loss(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.unwrap_model.compute_loss(outputs, targets)
        return loss
            
    @torch.no_grad()
    def eval(self):
        self.model.eval()
        correct = 0
        total = 0
        for batch_idx, batch in enumerate(self.eval_dataloader):
            inputs, targets = batch
            predicts = self.unwrap_model.predict(inputs)
            correct += (predicts == targets).sum()
            total += torch.tensor(inputs.size(0), device=self.accelerator.device)
        
        correct = self.accelerator.reduce(correct, 'sum').item()
        total = self.accelerator.reduce(total, 'sum').item()
        accuracy = correct / total
        self.accelerator.print(f"Accuracy {accuracy:.2%}")
        return accuracy
    
    @torch.no_grad()
    def _eval_epoch(self, epoch):
        accuracy = self.eval()
        self.accelerator.log({'accuracy': accuracy}, step=epoch)

    @torch.no_grad()
    def sample(self):
        self.model.eval()
        sample_points = torch.rand(self.config.sample.size, 2).to(self.accelerator.device)
        
        predicts = self.unwrap_model.predict(sample_points)
        
        if self.accelerator.is_main_process:
            cmap = {0: "r", 1: "g", 2: "b"}
            fig, ax = plt.subplots()
            data = sample_points.cpu()
            x = data[:, 0].clone()
            y = data[:, 1].clone()
            label = predicts.clone().cpu().tolist()
            ax.scatter(x, y, c=[cmap[l] for l in label], marker=".")
            plt.savefig(f"{self.sample_dir}/sample.png")