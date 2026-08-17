import torch 
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset

def get_loaders(batch_size=64):
    print("Loading Dataset...")

    #Transform: Translate the images to tensors and normalize them (between -1 and 1)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    #Download and load the CIFAR10 Dataset
    train_set = torchvision.datasets.CIFAR10(root='./data', train = True, download = True, transform = transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train = False, download = True, transform = transform)

    #Create the DataLoader for the training set
    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)

    return train_dataloader, test_loader


if __name__ =='__main__':
    train_loader, test_loader = get_loaders(batch_size= 64)
    images, labels = next(iter(train_loader))
    print(f"Batch Images Shapes: {images.shape}")