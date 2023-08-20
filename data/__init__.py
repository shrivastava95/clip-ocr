import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder

def check_determinism():
	x = [k for k in range(0, 50)]
	y = [k for k in range(0, 50)]
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
	assert (x_train == [48, 26, 25, 32, 19, 12, 4, 37, 8, 3, 6, 41, 46, 47, 15, 9, 16, 24, 34, 31, 0, 44, 27, 33, 5, 29, 11, 36, 1, 21, 2, 43, 35, 23, 40, 10, 22, 18, 49, 20, 7, 42, 14, 28, 38])

def train_val_dataset(dataset, val_split=0.25):
    check_determinism()
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, random_state=42)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets['train'], datasets['val']

def load_en_or(args, split, transform):
    dataset = ImageFolder('data/en_or/data', transform=transform)
    train_dataset, val_dataset = train_val_dataset(dataset, 0.25)
    if split == 'train':
        return train_dataset
    elif split == 'val':
        return val_dataset
    else:
        assert False, f"Not Implemented: The split '{split}' does not exist for this dataset. Check your implementation."

    
