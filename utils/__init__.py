import torch
from copy import deepcopy
import os
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from .training_methods import *
from .evaluation_methods import *

def save_checkpoint(args, model, epoch):
    torch.save(deepcopy(model.state_dict()), os.path.join(args.checkpoint_dir, str(epoch) + '.pt'))

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets['train'], datasets['val']

def make_classification_strings(args, class_names):
    if args.method == 'base':
        classification_strings = [f'image of {class_name} language text' for class_name in class_names] # prompt-engineered class template for ViT-B/16
    elif args.method == 'coop':
        classification_strings = ['X ' * args.n_ctx + class_name for class_name in class_names]
    elif args.method == 'cocoop':
        assert False, "Not Implemented: cocoop in utils/__init__.py"

def make_optimizer(args, model):
    if args.method == 'base':
        torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2) # taken from a paper.
    elif args.method == 'coop':
        torch.optim.Adam([model.trainable_param], lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2) # taken from a paper.
    elif args.method == 'cocoop':
        assert False, "Not Implemented: cocoop in utils/__init__.py"