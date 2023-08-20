import os
import argparse
import warnings
from PIL import Image, ImageDraw
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn
import torch
import clip
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tkinter import filedialog
from tqdm import tqdm

from utils import save_checkpoint
from utils import make_classification_strings
from utils import make_optimizer

from utils.training_methods import train_clip_base, train_clip_coop
from utils.evaluation_methods import accuracy_clip_base, accuracy_clip_coop

from utils import convert_models_to_fp32, convert_back_to_fp16

from data import load_en_or

warnings.simplefilter("ignore")

# TODO:
# 0. implement checkpoints - saving and loading in functions
# 1. implement the validation set wala stuff - if args.val_dir not None then: blabla
# 2. implement cosine annealing schedule with decay
# 3. implement learning rate warmup trick from 1e-5 (? the place where you read in coop it was with sgd)
# 4. in general, read clip finetuning papers and references to figure out best way to do this

def main(args):

    approach = {
        'method': args.method,
        'n_ctx': args.n_ctx,
    }
    model, preprocess = clip.load(args.clip_model,
                                  approach=approach,
                                  device=args.device, 
                                  jit=False)
    # dataset = ImageFolder(args.dataset_dir, transform=preprocess)
    # train_dataset, val_dataset = train_val_dataset(dataset, 0.25)
    if args.dataset == 'en_or':
        dataset = load_en_or(args, 'train', preprocess)
        train_dataset, val_dataset = load_en_or(args, 'train', preprocess), load_en_or(args, 'val', preprocess)
    else:
        assert False, f"Not Implemented: {args.dataset} dataset"

    train_loader, val_loader = DataLoader(train_dataset, args.batch_size, shuffle=True), DataLoader(val_dataset, args.batch_size, shuffle=True)
    classification_strings = make_classification_strings(args)
    
    if args.device == 'cpu':
        model.float()
    else:
        clip.model.convert_weights(model)  # actually this line is unnecessary since clip by default is already on float16

    optimizer = make_optimizer(args, model)
    criterion = nn.CrossEntropyLoss()

    losses = []
    train_accuracies = []
    val_accuracies = []

    if args.method == 'base':
        trainer, validator = train_clip_base, accuracy_clip_base
    elif args.method == 'coop':
        trainer, validator = train_clip_coop, accuracy_clip_coop
    elif args.method == 'cocoop':
        assert False, "Not implemented: cocoop trainer, validator in main.py"
    else:
        assert False, f"Not implemented: the method '{args.method}' is not yet implemented in main.py"

    if args.method == 'base':
        # no finetuning for whole clip. decide this later or make a seperate method called 'finetune'
        train_acc = validator(args, model, preprocess, train_loader, classification_strings) 
        val_acc = validator(args, model, preprocess, val_loader, classification_strings) 
        print(f'accuracy:  train:{train_accuracies[-1] * 100:.2f}   val:{val_accuracies[-1] * 100:.2f}')
    else:
        losses = losses + trainer(args, model, preprocess, train_loader, optimizer, criterion, classification_strings)
        train_acc = validator(args, model, preprocess, train_loader, classification_strings) 
        val_acc = validator(args, model, preprocess, val_loader, classification_strings) 
        train_accuracies.append(train_acc), val_accuracies.append(val_acc)
        print(f'before training:  train:{train_acc * 100:.2f}   val:{val_acc * 100:.2f}')
        for epoch in range(args.epochs):
            losses = losses + trainer(args, model, preprocess, train_loader, optimizer, criterion, classification_strings)
            train_acc = validator(args, model, preprocess, train_loader, classification_strings) 
            val_acc = validator(args, model, preprocess, val_loader, classification_strings) 
            train_accuracies.append(train_acc), val_accuracies.append(val_acc)
            print(f'end of epoch {epoch+1}:  train:{train_acc * 100:.2f}   val:{val_acc * 100:.2f}')
            if args.checkpoint_dir is not None:
                save_checkpoint(args, model, epoch)
        print(f'final accuracy:  train:{train_acc * 100:.2f}   val:{val_acc * 100:.2f}')

    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--method", type=str, default="coop", choices=["base", "coop", "cocoop"])
        parser.add_argument("--dataset", type=str, default="en_or", choices=["en_or"]) # can add your own in data/__init__.py
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--clip-model", type=str, default="ViT-B/16") # RN50x16, ViT-B/16
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument("--epochs", type=int, default=10) # should be around 50-200? idk. check some reference papers on finetuning clip.
        parser.add_argument("--lr", type=float, default=3e-4) # low might be because of initial learning rate explosion. finetune the transformers onto this.
        parser.add_argument("--n-ctx", type=int, default=16) # number of learned context embeddings for coop
        parser.add_argument("--checkpoint-dir", type=str, default='checkpoints') # if not None, saves a model for every epoch here.
        parser.add_argument("--load-from-checkpoint", type=str, default=None)
        args = parser.parse_args()
        return args

    args = get_args()
    main(args)
