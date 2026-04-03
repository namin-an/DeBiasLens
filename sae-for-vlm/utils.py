from pydoc import text
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import ImageNet, ImageFolder
import torch.nn as nn
from models.clip import Clip
from models.dino import Dino
from models.siglip import Siglip
import os
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from datasets import load_dataset, concatenate_datasets
from torchvision.datasets import CelebA
import torch
import pandas as pd

def get_collate_fn(processor, probe_text_enc):
    if probe_text_enc:
        def collate_fn(batch):
            texts = [txt[0] for txt in batch]
            return processor(texts, return_tensors="pt", padding='max_length', truncation=True)
    else:
        def collate_fn(batch):
            images = [img[0] for img in batch]
            return processor(images=images, return_tensors="pt", padding=True)
    return collate_fn

class CocoGenderTxtDataset(Dataset):
    def __init__(self, root, split="train"):
        self.df = pd.read_csv(os.path.join(root, f'{split}.csv'))
        self.captions = self.df['caption'].values
        self.genders = self.df['gender'].values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        txt = self.captions[idx]
        label = self.genders[idx]
        return txt, label


class BiosinBiasDataset(Dataset):
    def __init__(self, root, split="train"):
        self.ds = load_dataset("LabHC/bias_in_bios", split=split, cache_dir=root)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]
        txt = example["hard_text"]
        label = example["gender"]
        return txt, label

class FairFaceDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.ds = load_dataset("HuggingFaceM4/FairFace", "0.25", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]
        img = example["image"]  # already a PIL Image
        label = example["gender"]
        if self.transform:
            img = self.transform(img)
        return img, label
    
# class CelebAMy(Dataset):
#     """
#     Custom wrapper for the CelebA dataset.
    
#     Combines multiple attribute labels into a comma-separated string
#     to use as the text component for CLIP.
    
#     Args:
#         root (str): Root directory for CelebA data
#         split (str): Dataset split ('train', 'valid', or 'test')
#         **kwargs: Additional arguments to pass to CelebA constructor
#     """
#     def __init__(self, root, split='train', **kwargs):
#         self.celeba = CelebA(root, split=split, **kwargs)
#         self.attr_names = self.celeba.attr_names[:40]  # Using first 40 attributes
#         print(self.attr_names)
    
#     def __getitem__(self, index):
#         """Yield image samples with concatenated attribute labels as text."""
#         sample, target = self.celeba[index]
#         if self.transform:
#             sample = self.transform(sample)
#         # Get the indices of attributes that are True for this sample
#         labels_by_target = torch.nonzero(target)[:, 0]
#         # Convert attribute indices to attribute names and join with commas
#         target = ','.join([str(self.attr_names[x]) for x in labels_by_target])
#         return sample, target
    
#     def __len__(self):
#         """Return the number of samples in the dataset."""
#         return len(self.celeba)



class SBBench(Dataset):
    def __init__(self, is_crop, transform):
        if is_crop:
            crop_name = 'True'
        else:
            crop_name = 'False'

        self.transform = transform
        self.ds1 = load_dataset(f"vlmbias/sbbench_synthetic_gender_crop_{crop_name}")['train']
        self.ds2 = load_dataset(f"vlmbias/sbbench_synthetic_age_crop_{crop_name}")['train']
        self.ds = concatenate_datasets([self.ds1, self.ds2])
       

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]
        img = example["image"]
        if self.transform:
            img = self.transform(img)
        label = example["label"]
        return img, label


def get_dataset(args, preprocess, processor, split, subset=1.0):
    if args.dataset_name == 'cc3m':
        # if subset < 1.0:
        #     raise NotImplementedError
        # return get_cc3m(args, preprocess, split)
        raise NotImplementedError
    elif args.dataset_name == 'inat_birds':
        ds = ImageFolder(root=os.path.join(args.data_path, split), transform=preprocess)
    elif args.dataset_name == 'inat':
        ds = ImageFolder(root=os.path.join(args.data_path, split), transform=preprocess)
    elif args.dataset_name == 'imagenet':
        ds = ImageNet(root=args.data_path, split=split, transform=preprocess)
    elif args.dataset_name == 'cub':
        ds = ImageFolder(root=os.path.join(args.data_path, split), transform=preprocess)
    elif args.dataset_name == 'cocogender':
        ds = ImageFolder(root=os.path.join(args.data_path, split), transform=preprocess)
    elif args.dataset_name == 'fairface':
        ds = FairFaceDataset(split='train', transform=preprocess)
    elif args.dataset_name == 'celeba':
        ds = ImageFolder(root=args.data_path, transform=preprocess)
    elif args.dataset_name == 'bios':
        ds = BiosinBiasDataset(root=args.data_path)
    elif args.dataset_name == 'cocogendertxt':
        ds = CocoGenderTxtDataset(root=args.data_path)
    elif args.dataset_name == 'sbbench-syn':
        ds = SBBench(False, preprocess)
    elif args.dataset_name == 'sbbench-syn-crop':
        ds = SBBench(True, preprocess)

    keep_every = int(1.0 / subset)
    if keep_every > 1:
        ds = Subset(ds, list(range(0, len(ds), keep_every)))
    if processor is not None:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=get_collate_fn(processor, args.probe_text_enc))
    else:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return ds, dl

def get_dataset_only(dataset_name, data_path, preprocess=None, processor=None, split='train', subset=1.0):
    if dataset_name == 'cc3m':
        # if subset < 1.0:
        #     raise NotImplementedError
        # return get_cc3m(args, preprocess, split)
        raise NotImplementedError
    elif dataset_name == 'inat_birds':
        ds = ImageFolder(root=os.path.join(data_path, split), transform=preprocess)
    elif dataset_name == 'inat':
        ds = ImageFolder(root=os.path.join(data_path, split), transform=preprocess)
    elif dataset_name == 'imagenet':
        ds = ImageNet(root=data_path, split=split, transform=preprocess)
    elif dataset_name == 'cub':
        ds = ImageFolder(root=os.path.join(data_path, split), transform=preprocess)
    elif dataset_name == 'cocogender':
        ds = ImageFolder(root=os.path.join(data_path, split), transform=preprocess)
    elif dataset_name == 'fairface':
        ds = FairFaceDataset(split='train', transform=preprocess)
    elif dataset_name == 'celeba':
        ds = ImageFolder(root=data_path, transform=preprocess)
    elif dataset_name == 'bios':
        ds = BiosinBiasDataset(root=data_path)
    elif dataset_name == 'cocogendertxt':
        ds = CocoGenderTxtDataset(root=data_path)
    elif dataset_name == 'sbbench-syn':
        ds = SBBench(False, preprocess)
    elif dataset_name == 'sbbench-syn-crop':
        ds = SBBench(True, preprocess)

    return ds

def get_model(args):
    if args.model_name.startswith('clip'):
        clip = Clip(args.model_name, args.device, args.probe_text_enc)
        return clip, clip.processor
    elif args.model_name.startswith('dino'):
        dino = Dino(args.model_name, args.device, args.probe_text_enc)
        return dino, dino.processor
    elif args.model_name.startswith('siglip'):
        siglip = Siglip(args.model_name, args.device, args.probe_text_enc)
        return siglip, siglip.processor
    elif args.model_name.startswith('InternViT'):
        from models.internlm import InternViT
        internlm = InternViT(args.model_name, args.device)
        return internlm, internlm.processor
    elif args.model_name.startswith('llava'):
        from models.llavaonevision import LlavaOneVision
        llavaonevision = LlavaOneVision(args.model_name, args.device)
        return llavaonevision, llavaonevision.processor

def get_text_model(args):
    if args.model_name.startswith('clip'):
        model = CLIPTextModelWithProjection.from_pretrained(f"openai/{args.model_name}").to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(f"openai/{args.model_name}")
        return model, tokenizer

class IdentitySAE(nn.Module):
    def encode(self, x):
        return x
    def decode(self, x):
        return x
