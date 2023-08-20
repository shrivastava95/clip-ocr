import torch
from copy import deepcopy
import os
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from .training_methods import *
from .evaluation_methods import *


def save_checkpoint(args, model, epoch):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if args.method == 'base':
        pass
    
    elif args.method == 'coop':
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, str(epoch) + '.pt'))

    elif args.method == 'cocoop':
        assert False, "Not Implemented"

    else:
        assert False, "Not Implemented"

    torch.save(deepcopy(model.state_dict()), os.path.join(args.checkpoint_dir, str(epoch) + '.pt'))


def make_classification_strings(args):
    if args.method == 'base':
        class_names = ['english', 'odiya']
        classification_strings = [f'image of {class_name} language text' for class_name in class_names] # prompt-engineered class template for ViT-B/16

    elif args.method == 'coop':
        class_names = ['english', 'odiya']
        classification_strings = ['X ' * args.n_ctx + class_name for class_name in class_names]

    elif args.method == 'cocoop':
        assert False, "Not Implemented: cocoop in utils/__init__.py"

    else:
        assert False, f"Not implemented: the method '{args.method}' is not yet implemented in utils.py"

    return classification_strings


def make_optimizer(args, model):
    if args.method == 'base':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2) # taken from a paper.

    elif args.method == 'coop':
        optimizer = torch.optim.Adam([model.trainable_param], lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2) # taken from a paper.

    elif args.method == 'cocoop':
        assert False, "Not Implemented: cocoop in utils/__init__.py"

    else:
        assert False, f"Not implemented: the method '{args.method}' is not yet implemented in utils.py"

    return optimizer