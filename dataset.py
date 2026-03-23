import os
from torch.utils_data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image

def get_transforms(img_size=224):
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_test_transforms

def create_dataloaders(data_dir, batch_size=32, val_split=0.2):
    # Using ImageFolder from torchvision for convenience
    full_dataset = datasets.ImageFolder(data_dir)
    
    # Split dataset
    from sklearn.model_selection import train_test_split
    import torch
    
    train_idx, val_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=val_split,
        shuffle=True,
        stratify=full_dataset.targets
    )
    
    train_transforms, val_transforms = get_transforms()
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    # Applying transforms to subsets is a bit tricky with ImageFolder
    # so we'll wrap it in a custom class
    
    class TransformedSubset(Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
            
        def __len__(self):
            return len(self.subset)
            
    train_loader = DataLoader(
        TransformedSubset(train_dataset, train_transforms),
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        TransformedSubset(torch.utils.data.Subset(full_dataset, val_idx), val_transforms),
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, full_dataset.classes

if __name__ == "__main__":
    # Test dataloader creation
    # train_loader, val_loader, classes = create_dataloaders('dataset')
    pass
