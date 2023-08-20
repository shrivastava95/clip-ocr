import torch
import clip
from tqdm import tqdm

def accuracy_clip_base(args, model, preprocess, loader, classification_strings):
    correct, total = 0, 0
    with torch.no_grad():
      for batch in tqdm(loader):
          images, labels = batch
          images, labels = images.to(args.device), labels.to(args.device)
          texts = clip.tokenize(classification_strings).to(args.device)


          # logits_per_image, logits_per_text = model(images, texts)
          # get image features
          image_features = model.encode_image(images)
          image_features /= image_features.norm(dim=-1, keepdim=True)

          # get text features
          class_features = model.encode_text(clip.tokenize(classification_strings).to(args.device)) # using the modded text encoder
          similarity = (100 * image_features @ class_features.T).softmax(dim=-1)
          predictions = torch.argmax(similarity, dim=1)
          total += similarity.shape[0]
          correct += int(sum(predictions == labels))

    return correct / total

def accuracy_clip_coop(args, model, preprocess, loader, classification_strings):
    correct, total = 0, 0
    with torch.no_grad():
      for batch in tqdm(loader):
          images, labels = batch
          images, labels = images.to(args.device), labels.to(args.device)
          texts = clip.tokenize(classification_strings).to(args.device)


          # logits_per_image, logits_per_text = model(images, texts)
          # get image features
          image_features = model.encode_image(images)
          image_features /= image_features.norm(dim=-1, keepdim=True)

          # get text features
          class_features = model.encode_text_coop(clip.tokenize(classification_strings).to(args.device)) # using the modded text encoder
          similarity = (100 * image_features @ class_features.T).softmax(dim=-1)
          predictions = torch.argmax(similarity, dim=1)
          total += similarity.shape[0]
          correct += int(sum(predictions == labels))

    return correct / total