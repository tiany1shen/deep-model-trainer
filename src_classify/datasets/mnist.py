import torchvision

class MNIST(torchvision.datasets.MNIST):
    def __init__(self, args):
        root = args.root 
        train = args.train
        transforms = self.transforms()
        
        super().__init__(root, train=train, transform=transforms)
    
    def transforms(self):
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5))
        ])
    
    def inv_transforms(self):
        return torchvision.transforms.Normalize((-1.0,), (2.0,))


