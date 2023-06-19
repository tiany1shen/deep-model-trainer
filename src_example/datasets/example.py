import torch 
import json
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from easydict import EasyDict
from pathlib import Path

class ExampleDataset(Dataset):
    def __init__(self, args: EasyDict):
        self.num_classes = 3
        if hasattr(args, 'data_path') and Path(args.data_path).exists():
            with open(Path(args.data_path), 'r') as f:
                data_dict = json.load(f)
            assert "data" in data_dict, "data should be a key in the json file when specifying data_path"
            self.data = torch.tensor(data_dict["data"])
            if "labels" in data_dict:
                self.label = torch.tensor(data_dict["labels"])
            else:
                self.label = None
            self.size = len(self.data)
        else:
            assert hasattr(args, 'size'), "size should be specified when data_path is not specified"
            self.size = args.size
            seed = args.seed if hasattr(args, 'seed') else None 
            self.data = self._generate_data(seed)
            self.label = None
        
        if self.label is None:
            self.label = self._calculate_label()
        if hasattr(args, 'one_hot') and args.one_hot:
            self.label = one_hot(self.label, self.num_classes)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx: int):
        return self.data[idx], self.label[idx]
    
    def _generate_data(self, seed: int = None):
        if seed is not None:
            torch.manual_seed(seed)
        return torch.rand(self.size, 2)
    
    def _calculate_label(self):
        label_0 = ((self.data[:, 0] > 0.5) & (self.data[:, 1] > 0.5)).long()
        label_1 = (self.data[:, 0] ** 2 + self.data[:, 1] ** 2 < 0.5).long()
        return 2 - 2 * label_0 - label_1