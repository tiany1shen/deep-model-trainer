import torch
from .Trainer import Trainer 


class ClassifyTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        
    def _compute_loss(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.unwrap_model.compute_loss(outputs, targets)
        return {"loss": loss}, {"loss": 1.0}
    
    def _register_custom_metrics(self):
        self.tracker.register(
            ["loss", "accuracy"]
        )
        
    def _log_metrics(self, epoch):
        losses = self.tracker.fetch("accuracy", reductions='last')
        self.accelerator.log(losses, step=epoch)
    
    @torch.no_grad()
    def eval(self):
        self.model.eval()
        correct = 0
        total = 0
        for batch_idx, (imgs, labels) in enumerate(self.eval_dataloader):
            predicts = self.unwrap_model.predict(imgs, is_logit=False)
            correct += (predicts == labels).sum()
            total += torch.tensor(imgs.size(0), device=labels.device)
            
        correct = self.accelerator.reduce(correct, 'sum').item()
        total = self.accelerator.reduce(total, 'sum').item()
        accuracy = correct / total
        self.accelerator.print(f"Accuracy {accuracy:.2%}")
        self.tracker.update("accuracy", torch.Tensor([accuracy]))
        
    @torch.no_grad()
    def _eval_epoch(self, epoch):
        self.eval()