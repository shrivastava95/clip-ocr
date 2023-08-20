# clip-ocr
A repository for organizing contributions related to finetuning and related approaches using OpenAI's CLIP for the Dictionary Augmented Translations project in C4GT'23.

# Instructions
Run `main.py` with the proper arguments. Here are some details and the various options related to the arguments:
- `--method`: either "base" (without training) or "coop" or "cocoop" (these are based on https://github.com/KaiyangZhou/CoOp).
- `--dataset-dir`: a directory to read the dataset from. It should be in standard imagefolder format. Implementation of a dataset loader is scheduled for later.
- `--device`: the device used for training the models
- `--clip-model`: the clip model used. For example, "RN50x16" or "ViT-B/16". You can use any of the officially supported OpenAI-CLIP models.
- `--batch-size`: the batch size used by the dataloader.
- `--epochs`: number of epochs.
- `--lr`: the learning rate for finetuning or training.
- `--n-ctx`: the number of learnable context vectors. Refer to the CoOp paper / repository.
- `--checkpoint-dir`: the directory to which the models are saved to every epoch. They are saved as "checkpoint-dir/0.pt", "checkpoint-dir/1.pt"... and so on.

# Requirements
