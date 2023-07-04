from typing import overload, Optional
# from typing import override
from torch import Tensor
from collections.abc import Iterable
from enum import Enum


def to_list(value):
    if isinstance(value, list):
        return value
    else:
        return [value]

class Reduction(Enum):
    Id  = lambda x: x
    Mean= lambda x: sum(x) / len(x)
    Min = lambda x: min(x)
    Max = lambda x: max(x)
    Sum = lambda x: sum(x)
    Last= lambda x: x[-1]
    
    @classmethod
    def alias(self, reduction: Optional[str] = None):
        if reduction is None:
            return "Id"
        elif hasattr(Reduction, reduction):
            return reduction
        elif reduction in ['id', 'identity', 'ID', 'none', 'None']:
            return "Id"
        elif reduction in ['mean', 'avg', 'average']:
            return "Mean"
        elif reduction in ['min', 'minimum', 'smallest']:
            return "Min"
        elif reduction in ['max', 'maximum', 'largest', 'biggest']:
            return "Max"
        elif reduction in ['sum', 'summary', 'total']:
            return "Sum"
        elif reduction in ['last', 'final', 'end', 'tail']:
            return "Last"
        else:
            raise ValueError(f"Unknown reduction {reduction}")
    
    @classmethod    
    def get(self, reduction: Optional[str] = None):
        return getattr(Reduction, self.alias(reduction))


ValueType = Tensor or list[Tensor] # Tensor must be of shape (1,)
OutputType = int or float or list[int or float]
    
class MetricTracker:
    def __init__(self):
        self.metrics = {}
    
    @overload
    def register(self, name: str or list[str]) -> None:...
    @overload
    def register(self, metric_dict: dict[str, ValueType]) -> None:...
    
    def register(self, args):
        """ 
        Register metrics to be tracked. 
        
        Args: str | List[str] | Dict[str, ValueType]
          If register by str or list, the value will be initialized as empty list.
          If register by dict, the value will be initialized as a list of dict[key].
        
          After register, `self.metrics` will be of type `dict[str, list[Tensor]]`.
          Involved key-value pairs in the original `self.metrics` will be replaced, and 
          those not involved will be preserved.
          
        Examples:
        
        register by name (str)
        >>> tracker = MetricTracker()
        >>> tracker.register('loss')
        >>> tracker.metrics
        {'loss': []}
        
        register by list of names (List[str])
        >>> tracker = MetricTracker()
        >>> tracker.register(['loss', 'accuracy'])
        >>> tracker.metrics
        {'loss': [], 'accuracy': []}
        
        register by dict of names and values (Dict[str, ValueType])
        >>> tracker = MetricTracker()
        >>> tracker.register({'loss': tensor(0.1), 'accuracy': [tensor(0.8), tensor(0.9)]})
        >>> tracker.metrics
        {'loss': [tensor(0.1)], 'accuracy': [tensor(0.8), tensor(0.9)]}
        
        reset and register
        >>> tracker.metrics = {"loss": [tensor(0.1)], "accuracy": [tensor(0.8), tensor(0.9)]}
        >>> tracker.register(['loss', 'perplexity'])
        >>> tracker.metrics
        {'loss': [], "accuracy": [tensor(0.8), tensor(0.9)], 'perplexity': []}
        
        """
        if isinstance(args, str):
            metric_dict = {args: []}
        if isinstance(args, list):
            metric_dict = {name: [] for name in args}
        if isinstance(args, dict):
            metric_dict = {name: to_list(value) for name, value in args.items()}
        self.metrics.update(metric_dict)
    
    @overload
    def update(self, name: str, value: ValueType) -> None:...
    @overload
    def update(self, names: list[str], values: list[ValueType]) -> None:...
    @overload
    def update(self, metric_dict: dict[str, ValueType]) -> None:...
    
    def update(self, *args):
        """ 
        Add new values to a existing named metric list.
        """
        if len(args) == 2:
            names, values = args
            if isinstance(names, str):
                names = [names]
                values = [values]
            update_dict = {name: to_list(value) for name, value in zip(names, values)}
        if len(args) == 1:
            update_dict = {name: to_list(value) for name, value in args[0].items()}
        for name, value_list in update_dict.items():
            assert name in self.metrics, f"Metric {name} not registered"
            self.metrics[name].extend(value_list)
    
    def fetch(self, names: str or list[str], reductions: Optional[str or list[str]] = None) -> dict[str, OutputType]:
        """ 
        fetch values stored in a named metric list. The `reduction` argument determines how to 
        reduce these values into specific type.
        
        """
        names = to_list(names)
        reductions = to_list(reductions)
        if len(reductions) == 1:
            different_reduction = False
            reduce_values = Reduction.get(reductions[0])
        else:
            different_reduction = True
            assert len(reductions) >= len(names), "Number of reductions must be no less than number of names"
            
        metrics = {}
        for name, reduction in zip(names, reductions):
            assert name in self.metrics, f"Metric {name} not registered"
            assert len(self.metrics[name]) > 0, f"Metric {name} has no value"
            
            self.metrics[name] = self.process_before_fetch(self.metrics[name])
            history_scalars = [metirc.item() for metirc in self.metrics[name]]
            
            if different_reduction:
                reduce_values = Reduction.get(reduction)
            metrics[name] = reduce_values(history_scalars)
        return metrics
    
    def process_before_fetch(self, tensor_list: list[Tensor]) -> list[Tensor]:
        return tensor_list
        


class SyncMetricTracker(MetricTracker):
    """ 
    MetricTracker use for multi-process scene
    
    """
    def __init__(self):
        super().__init__()
        
        from accelerator.utils import reduce as sync 
        self.sync = sync
    
    # @override # need Python >= 3.12
    def process_before_fetch(self, tensor_list: list[Tensor]) -> list[Tensor]:
        return self.sync(tensor_list)