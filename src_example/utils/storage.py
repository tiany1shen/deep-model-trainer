from accelerate.utils import reduce


class Storage:
    def __init__(self):
        self.metrics = {}
    
    def register(self, name: str):
        if name not in self.metrics:
            self.metrics[name] = []
    
    def reset(self, name: str):
        assert name in self.metrics, f"Metric {name} not registered"
        self.metrics[name] = []
    
    def update(self, name: str, value: float):
        self.register(name)
        self.metrics[name].append(value)
        
    def get(self, name: str, reduction: str=None, **kwargs):
        assert name in self.metrics, f"Metric {name} not registered"
        assert len(self.metrics[name]) > 0, f"Metric {name} has no value"
        assert reduction in [None, 'mean', 'max', 'min', 'last', 'sum'], f"Reduction {reduction} not supported"
        
        history_metircs = reduce(self.metrics[name])
        history_metircs = [metirc.item() for metirc in history_metircs]
        
        if reduction is None:
            return history_metircs
        
        if reduction == 'mean':
            return sum(history_metircs) / len(history_metircs)
        
        if reduction == 'max':
            return max(history_metircs)
        
        if reduction == 'min':
            return min(history_metircs)
        
        if reduction == 'last':
            return history_metircs[-1]
        
        if reduction == 'sum':
            return sum(history_metircs)

        