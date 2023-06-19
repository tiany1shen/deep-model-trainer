# datasets & models

README in [English](/README.md) | [简体中文](/readme/README_zh_CN.md)

`datasets` and `models` package provide dataset interfaces inherited from `torch.utils.data.Dataset` and model interfaces inherited from `torch.nn.Module`. All these interfaces should be able to instantiate by only one argument `args` or `configs`.

## dataset example

For example, if you want to load data from directories of json file, you can write a dataset class like this:

```python
class JsonDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.dirs = args.dirs
        self._scan_data()

    def _scan_data(self):
        self.paths = []
        for dir in self.dirs:
            self.data += os.listdir(dir)

    def __getitem__(self, index):
        path = self.paths[index]
        with open(path, 'r') as f:
            data = json.load(f)
        # assume each json file has a key named "data" and its value is a list of numbers
        return torch.Tensor(data["data"])

    def __len__(self):
        return len(self.paths)
```

We restrict the users can only access to the dataset objects in `datasets` package, so you should register your dataset class in `datasets/__init__.py`:

```python
""" In models/__init__.py """
from .JsonDataset import JsonDataset
from .ExampleDataset import ExampleDataset

__all__ = ['JsonDataset','ExampleDataset']
```

## model example

To build an MLP model, you can write a model class like this:

```python
from torch import nn

args = EasyDict({
    "input_dim": 2,
    "hidden_dims": [16, 64, 32],
    "output_dim": 3
})

class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_dims = [args.input_dim] + args.hidden_dims
        self.output_dims = args.hidden_dims + [args.output_dim]

        layers = []
        for i in range(len(self.input_dims)):
            layers.append(nn.Linear(self.input_dims[i], self.output_dims[i]))
            if i != len(self.input_dims) - 1:
                layers.append(nn.ReLU())
        layers.append(nn.Softmax(dim=-1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
```
