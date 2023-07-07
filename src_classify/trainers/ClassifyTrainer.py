import torch
from .Trainer import Trainer 


class ClassifyTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        
    def _compute_loss(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.unwrap_model.compute_loss(outputs, targets)
        return {"CrossEntropy_Loss": loss}, dict(zip(self.loss_names, self.loss_weights))
        
    @torch.no_grad()
    def _eval_epoch(self, epoch):
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
        self.tracker.update("Accuracy_Metric", torch.Tensor([accuracy]))