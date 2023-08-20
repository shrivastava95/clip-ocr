import torch
import os
import clip
from tqdm import tqdm
from torch import nn

# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

def train_clip_base(args, model, preprocess, loader, optimizer, criterion, classification_strings):
    for batch in tqdm(loader):
        optimizer.zero_grad()
        images, labels = batch
        images, labels = images.to(args.device), labels.to(args.device)
        texts = clip.tokenize(classification_strings).to(args.device)

        # get image features
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # get text features
        class_features = model.encode_text(texts) # using the modded text encoder
        similarity = (100 * image_features @ class_features.T).softmax(dim=-1)
        predictions = torch.argmax(similarity, dim=1)

        loss = nn.CrossEntropyLoss()(similarity, labels)
        loss.backward()
        if args.device == "cpu":
            optimizer.step()
        else :
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

def train_clip_coop(args, model, preprocess, loader, optimizer, criterion, classification_strings):
    class_texts_template = ['X ' * args.n_ctx + class_name for class_name in classification_strings]
    for batch in tqdm(loader):
        optimizer.zero_grad()
        images, labels = batch
        images, labels = images.to(args.device), labels.to(args.device)
        texts = clip.tokenize(classification_strings).to(args.device)

        # get image features
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # get text features
        class_features = model.encode_text_coop(texts) # using the modded text encoder
        similarity = (100 * image_features @ class_features.T).softmax(dim=-1)
        predictions = torch.argmax(similarity, dim=1)

        loss = nn.CrossEntropyLoss()(similarity, labels)
        loss.backward()
        if args.device == "cpu":
            optimizer.step()
        else :
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)