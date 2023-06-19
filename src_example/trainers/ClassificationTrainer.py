import torch 
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
