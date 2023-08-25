import torch
import os
import clip
from tqdm import tqdm
from torch import nn


# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
          p.data = p.data.float()
          if p.grad is not None:
              p.grad.data = p.grad.data.float()


def convert_back_to_fp16(args, model):
    if args.method == 'base':
        clip.model.convert_weights(model)
    elif args.method == 'coop':
        clip.model.convert_weights(model)
        model.trainable_param.data = model.trainable_param.data.to(torch.float16)
    else:
        assert False, "Not Implemented"


def train_clip_base(args, model, preprocess, loader, optimizer, criterion, classification_strings):
    losses = []
    for batch in tqdm(loader):
        optimizer.zero_grad()
        images, labels = batch
        images, labels = images.to(args.device), labels.to(args.device)
        texts = clip.tokenize(classification_strings).to(args.device)

        # get image features
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # get text features
        class_features = model.encode_text(texts)
        class_features = class_features / class_features.norm(dim=-1, keepdim=True)

        # compute similarity
        similarity = (100 * image_features @ class_features.T).softmax(dim=-1)
        predictions = torch.argmax(similarity, dim=1)

        loss = nn.CrossEntropyLoss()(similarity, labels)
        loss.backward()
        losses.append(loss.item())
        if args.device == "cpu":
            optimizer.step()
        else :
            convert_models_to_fp32(model)
            optimizer.step()
            convert_back_to_fp16(args, model)
    return losses

def train_clip_coop(args, model, preprocess, loader, optimizer, criterion, classification_strings):
    losses = []
    for batch in tqdm(loader):
        optimizer.zero_grad()
        images, labels = batch
        images, labels = images.to(args.device), labels.to(args.device)
        texts = clip.tokenize(classification_strings).to(args.device)

        # get image features
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # get text features
        class_features = model.encode_text_coop(texts) # using the modded text encoder
        class_features = class_features / class_features.norm(dim=-1, keepdim=True)

        # compute similarity
        similarity = (100 * image_features @ class_features.T).softmax(dim=-1)
        predictions = torch.argmax(similarity, dim=1)

        loss = nn.CrossEntropyLoss()(similarity, labels)
        loss.backward()
        losses.append(loss.item())
        if args.device == "cpu":
            optimizer.step()
        else :
            convert_models_to_fp32(model)
            optimizer.step()
            convert_back_to_fp16(args, model)
    return losses

def train_clip_cocoop(args, model, preprocess, loader, optimizer, criterion, classification_strings):
    losses = []
    for batch in tqdm(loader):
        optimizer.zero_grad()
        images, labels = batch
        images, labels = images.to(args.device), labels.to(args.device)
        texts = clip.tokenize(classification_strings).to(args.device)

        # get image features
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # get text features
        class_features = model.encode_text_cocoop(texts, image_features) # using the modded text encoder
        class_features = class_features / class_features.norm(dim=-1, keepdim=True)

        # compute similarity
        image_features = image_features.unsqueeze(2)
        similarity = torch.bmm(class_features, 100 * image_features).softmax(dim=-1)

        loss = nn.CrossEntropyLoss()(similarity, labels)
        loss.backward()
        losses.append(loss.item())
        if args.device == "cpu":
            optimizer.step()
        else :
            convert_models_to_fp32(model)
            optimizer.step()
            convert_back_to_fp16(args, model)
    return losses
