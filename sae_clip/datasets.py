import os
import subprocess
import pathlib
from abc import ABC
from typing import Callable, Union
import requests
from io import BytesIO
import random

import csv
import numpy as np
import pandas as pd
import torch
from PIL import Image
from gdown import download
from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision.datasets import ImageFolder



from sae_clip import Dotdict, FAIRFACE_DATA_PATH


class IATDataset(Dataset, ABC):
    GENDER_ENCODING = {"Female": 1, "Male": 0}
    # AGE_ENCODING = {"0-2": 0, "3-9": 1, "10-19": 2, "20-29": 3, "30-39": 4,
    #                 "40-49": 5, "50-59": 6, "60-69": 7, "more than 70": 8}
    AGE_ENCODING = {"3-9": 0, "10-19": 1, "20-29": 2, "30-39": 3,
                    "40-49": 4, "50-59": 5, "60-69": 6} #, "more than 70": 8}
    
    GENDER_ENCODING_PAIRS = {"woman": 1, "man": 0}
    GENDER_ENCODING_PATA = {"female": 1, "male": 0}
    AGE_ENCODING_PATA = {"young": 0, "old": 1}

    LABEL_ENCODING_PATA = {0: 'tench',
                           1: 'English springer',
                           2: 'cassette player',
                           3: 'chain saw',
                           4: 'church',
                           5: 'French horn',
                           6: 'garbage truck',
                           7: 'gas pump',
                           8: 'golf ball',
                           9: 'parachute'}

    def __init__(self):
        self.image_embeddings: torch.Tensor = None
        self.iat_labels: np.ndarray = None
        self._img_fnames = None
        self._transforms = None
        self.use_cache = None
        self.iat_type = None
        self.n_iat_classes = None

    def gen_labels(self, iat_type: str, label_encoding: object = None):
        # WARNING: iat_type == "pairwise_adjective" is no longer supported
        if iat_type in ("gender_science", "test_weat", "gender"):
            labels_list = self.labels["gender"]
            label_encoding = IATDataset.GENDER_ENCODING if label_encoding is None else label_encoding
        elif iat_type == "race":
            labels_list = self.labels["race"]
            label_encoding = self.RACE_ENCODING if label_encoding is None else label_encoding
        elif iat_type == "age":
            labels_list = self.labels["age"]
            label_encoding = IATDataset.AGE_ENCODING if label_encoding is None else label_encoding
        else:
            raise NotImplementedError
        assert set(labels_list.unique()) == set(label_encoding.keys()), "There is a missing label, invalid for WEAT"
        labels_list = np.array(labels_list.apply(lambda x: label_encoding[x]), dtype=int)
        # assert labels_list.sum() != 0 and (1 - labels_list).sum() != 0, "Labels are all equal, invalid for Weat"
        return labels_list, len(label_encoding)

    def gen_labels_pairs(self, iat_type: str, label_encoding: object = None):
        # WARNING: iat_type == "pairwise_adjective" is no longer supported
        if iat_type in ("gender_science", "test_weat", "gender"):
            labels_list = self.labels["gender"]
            label_encoding = IATDataset.GENDER_ENCODING_PAIRS if label_encoding is None else label_encoding
        elif iat_type == "race":
            labels_list = self.labels["race"]
            label_encoding = self.RACE_ENCODING_PAIRS if label_encoding is None else label_encoding
        else:
            raise NotImplementedError
        assert set(labels_list.unique()) == set(label_encoding.keys()), "There is a missing label, invalid for WEAT"
        labels_list = np.array(labels_list.apply(lambda x: label_encoding[x]), dtype=int)
        return labels_list, len(label_encoding)

    def gen_labels_pata(self, iat_type: str, label_encoding: object = None):
        # WARNING: iat_type == "pairwise_adjective" is no longer supported
        if iat_type in ("gender_science", "test_weat", "gender"):
            labels_list = self.labels["gender"]
            label_encoding = IATDataset.GENDER_ENCODING_PATA if label_encoding is None else label_encoding
        elif iat_type == "race":
            labels_list = self.labels["race"]
            label_encoding = self.RACE_ENCODING_PATA if label_encoding is None else label_encoding
        elif iat_type == "age":
            labels_list = self.labels["age"]
            label_encoding = IATDataset.AGE_ENCODING_PATA if label_encoding is None else label_encoding
        else:
            raise NotImplementedError
        assert set(labels_list.unique()) == set(label_encoding.keys()), "There is a missing label, invalid for WEAT"
        labels_list = np.array(labels_list.apply(lambda x: label_encoding[x]), dtype=int)
        return labels_list, len(label_encoding)
    
    def gen_labels_imagenet(self, label_encoding: object = None):
        labels_list = self.labels["label"]
        label_encoding = IATDataset.LABEL_ENCODING_PATA if label_encoding is None else label_encoding
        assert set(labels_list.unique()) == set(label_encoding.keys()), "There is a missing label, invalid for WEAT"
        return labels_list, len(label_encoding)



class FairFace(IATDataset):
    RACE_ENCODING = {"White": 0, "Southeast Asian": 1, "Middle Eastern": 2,
                     "Black": 3, "Indian": 4, "Latino_Hispanic": 5, "East Asian": 6}

    def __init__(self, iat_type: str = None, lazy: bool = True, mode: str = "train",
                 _n_samples: Union[float, int] = None, transforms: Callable = None, equal_split: bool = True, is_age: bool = False):
        self.DATA_PATH = str(FAIRFACE_DATA_PATH)
        self.download_data()
        self.mode = mode
        self._transforms = (lambda x: x) if transforms is None else transforms
        self.labels = pd.read_csv(os.path.join(self.DATA_PATH, "labels", mode, f"{mode}_labels.csv"))
        self.labels.sort_values("file", inplace=True)
        if _n_samples is not None:
            if isinstance(_n_samples, float):
                _n_samples = int(len(self.labels) * _n_samples)

            self.labels = self.labels[:_n_samples]
        if equal_split:

            ##############
            # labels_male = self.labels.loc[self.labels['gender'] == 'Male']
            # labels_female = self.labels.loc[self.labels['gender'] == 'Female']

            # num_females = labels_female.count()[0]
            # num_males = labels_male.count()[0]

            # sample_num = min(num_males, num_females)

            # labels_male = labels_male.sample(n=sample_num, random_state=1)
            # labels_female = labels_female.sample(n=sample_num, random_state=1)

            # self.labels = pd.concat([labels_male, labels_female], ignore_index=True) #labels_male.append(labels_female, ignore_index=True)
            # self.labels.to_csv('fairface_labels.csv', index=False) 
            #################

            if is_age:
                exclude_ages = ["more than 70", "0-2"]
                self.labels = self.labels[~self.labels["age"].isin(exclude_ages)].reset_index(drop=True)

            ################# - rebuttal
            # balanced = []

            # group_cols = ['age', 'race']

            # for (_, _), group in self.labels.groupby(group_cols):
            #     gender_counts = group['gender'].value_counts()

            #     # only keep groups where both genders exist
            #     if {'Male', 'Female'}.issubset(gender_counts.index):
            #         n = gender_counts.min()

            #         sampled = (
            #             group
            #             .groupby('gender', group_keys=False)
            #             .sample(n=n, random_state=1)
            #         )
            #         balanced.append(sampled)

            # self.labels = pd.concat(balanced, ignore_index=True)
            # self.labels.to_csv('fairface_labels_balanced.csv', index=False)
            #################
            

            # lst = [7734,  7818,  9745,  8072,  9712,  8707,  8404,  5958, 10045, 3154]
            # lst = [4044, 2863, 1948, 5988,  200, 3552, 2857, 6832,  120, 3408]
            # result = []
            # for l in lst:
            #      result.append(self.labels.iloc[l]['file'])
            # print(result)
            # print(hi)

        self._img_fnames = [os.path.join(self.DATA_PATH, "imgs", "train_val", x) for x in self.labels["file"]]
        self._fname_to_inx = {fname: inx for inx, fname in enumerate(self._img_fnames)}

        self.images_list = None
        if not lazy:
            self.images_list = list(map(self.__getitem__, range(len(self.labels))))

        self.iat_labels = self.gen_labels(iat_type=iat_type)[0]

    def download_data(self):
        os.makedirs(self.DATA_PATH, exist_ok=True)
        # Use 1.25 padding
        fairface_parts = {
            "imgs": {
                "train_val": ("https://drive.google.com/uc?id=1g7qNOZz9wC7OfOhcPqH1EZ5bk1UFGmlL", "train_val_imgs.zip"),
            },
            "labels": {
                "train": ("https://drive.google.com/uc?id=1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH", "train_labels.csv"),
                "val": ("https://drive.google.com/uc?id=1wOdja-ezstMEp81tX1a-EYkFebev4h7D", "val_labels.csv")
            }
        }

        for part_name, part in fairface_parts.items():
            for subpart_name, (subpart_url, subpart_fname) in part.items():
                subpart_dir = os.path.join(self.DATA_PATH, part_name, subpart_name)
                if os.path.isdir(subpart_dir):
                    continue
                os.makedirs(subpart_dir, exist_ok=True)
                print(f"Downloading fairface {subpart_name} {part_name}...")
                output_path = os.path.join(subpart_dir, subpart_fname)
                download(subpart_url, output=output_path)

                if subpart_fname.endswith(".zip"):
                    print(f"Unzipping {subpart_name} {part_name}...")
                    subprocess.check_output(["unzip", "-d", subpart_dir, output_path])
                    os.remove(output_path)
                    print(f"Done unzipping {subpart_name} {part_name}.")
                print(f"Done with fairface {subpart_name} {part_name}.")

    def _load_fairface_sample(self, sample_labels) -> dict:
        res = Dotdict(dict(sample_labels))
        img_fname = os.path.join(self.DATA_PATH, "imgs", "train_val", res.file)
        try:
            res.img = self._transforms(Image.open(img_fname))
        except:
            try:
                res.img = self._transforms(images=Image.open(img_fname))
            except: pass
        return res

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_fairface_sample(self.labels.iloc[index])
        ff_sample.iat_label = self.iat_labels[index]
        return ff_sample

    def __len__(self):
        return len(self.labels)
    

class Pairs(IATDataset):
    RACE_ENCODING_PAIRS = {"white": 0, "black": 1}

    def __init__(self, iat_type: str = None, lazy: bool = True, 
                 _n_samples: Union[float, int] = None, transforms: Callable = None, equal_split: bool = True, ):
        DATA_PATH = (pathlib.Path(__file__) / ".." / ".." / "data").resolve().absolute()
        self.DATA_PATH = DATA_PATH / 'PAIRS' / 'data'
        self._transforms = (lambda x: x) if transforms is None else transforms

        # List to store image paths and their corresponding emotions
        data = []

        # Loop through each folder
        for typ in os.listdir(self.DATA_PATH):
            folder_dir = os.path.join(self.DATA_PATH, typ)
            for typ2 in os.listdir(folder_dir):
                folder_dir2 = os.path.join(folder_dir, typ2)
                for image_name in os.listdir(folder_dir2):
                    if image_name.endswith('.jpg') or image_name.endswith('.png'):
                        image_path = os.path.join(folder_dir2, image_name)
        
                        image_file_name = os.path.basename(image_path)
                        race, gender = image_file_name.split('_')[0], image_file_name.split('_')[1].split('.')[0]
                        data.append([image_path, typ, typ2, race, gender])                        
        
        # Create a DataFrame
        self.labels = pd.DataFrame(data, columns=['file', 'meta1', 'meta2', 'race', 'gender'])

        self.labels.sort_values("file", inplace=True)
        if _n_samples is not None:
            if isinstance(_n_samples, float):
                _n_samples = int(len(self.labels) * _n_samples)

            self.labels = self.labels[:_n_samples]
        if equal_split:
            labels_male = self.labels.loc[self.labels['gender'] == 'man']
            labels_female = self.labels.loc[self.labels['gender'] == 'woman']

            num_females = labels_female.count()[0]
            num_males = labels_male.count()[0]

            sample_num = min(num_males, num_females)

            labels_male = labels_male.sample(n=sample_num, random_state=1)
            labels_female = labels_female.sample(n=sample_num, random_state=1)

            self.labels = pd.concat([labels_male, labels_female], ignore_index=True) #labels_male.append(labels_female, ignore_index=True)

        self._img_fnames = [x for x in self.labels["file"]]
        self._fname_to_inx = {fname: inx for inx, fname in enumerate(self._img_fnames)}

        self.images_list = None
        if not lazy:
            self.images_list = list(map(self.__getitem__, range(len(self.labels))))

        self.iat_labels = self.gen_labels_pairs(iat_type=iat_type)[0]

    def _load_pairs_sample(self, sample_labels) -> dict:
        res = Dotdict(dict(sample_labels))
        img_fname = res.file
        try:
            res.img = self._transforms(Image.open(img_fname))
        except:
            try:
                res.img = self._transforms(images=Image.open(img_fname))
            except: pass
        return res

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_pairs_sample(self.labels.iloc[index])
        ff_sample.iat_label = self.iat_labels[index]
        ff_sample.orig_labels_list = self._img_fnames[index]
        return ff_sample

    def __len__(self):
        return len(self.labels)


class PATA(IATDataset):
    RACE_ENCODING_PATA = {"black": 0, "caucasian": 1, "eastasian": 2, 'hispanic': 3, 'indian': 4}

    def __init__(self, iat_type: str = None, lazy: bool = True, 
                 _n_samples: Union[float, int] = None, transforms: Callable = None, equal_split: bool = True, ):
        DATA_PATH = (pathlib.Path(__file__) / ".." / ".." / "data").resolve().absolute()

        self.DATA_PATH = DATA_PATH / 'pata_dataset_orig' / 'images_tmp'  #/ 'pata_dataset'
        self.IMAGE_DATA_PATH = DATA_PATH / 'pata_dataset_orig' / 'pata_fairness.files.lst'
        with open(self.IMAGE_DATA_PATH, "r", encoding="utf-8") as f:
            image_list = [line.strip() for line in f if line.strip()]

        self._transforms = (lambda x: x) if transforms is None else transforms

        # Loop through each emotion folder
        for line in image_list:
            tag, url = line.strip().split("|", 1)
            parts = tag.split("_")  # e.g. forest_indian_male_young

            # Create a safe filename
            fname = f"{tag}.png"
            fpath = self.DATA_PATH / fname

            if not fpath.exists():  # avoid redownloading
                try:
                    resp = requests.get(url, timeout=10)
                    resp.raise_for_status()
                    img = Image.open(BytesIO(resp.content)).convert("RGB")
                    img.save(fpath)
                except Exception as e:
                    pass
            continue

        data = []

        for image_name in os.listdir(self.DATA_PATH):
            if image_name.endswith('.jpg') or image_name.endswith('.png'):
                image_path = os.path.join(self.DATA_PATH, image_name)
                parts = image_name.split('.')[0].split('_')
                context, race, gender, age = parts[0], parts[1], parts[2], parts[3]
                data.append([image_path, context, race, gender, age])

        # Create a DataFrame
        self.labels = pd.DataFrame(data, columns=['file', 'context', 'race', 'gender', 'age'])

        self.labels.sort_values("file", inplace=True)
        if _n_samples is not None:
            if isinstance(_n_samples, float):
                _n_samples = int(len(self.labels) * _n_samples)

            self.labels = self.labels[:_n_samples]
        if equal_split:
            labels_male = self.labels.loc[self.labels['gender'] == 'male']
            labels_female = self.labels.loc[self.labels['gender'] == 'female']

            num_females = labels_female.count()[0]
            num_males = labels_male.count()[0]

            sample_num = min(num_males, num_females)

            labels_male = labels_male.sample(n=sample_num, random_state=1)
            labels_female = labels_female.sample(n=sample_num, random_state=1)

            self.labels = pd.concat([labels_male, labels_female], ignore_index=True) 

        self._img_fnames = [x for x in self.labels["file"]]
        self._fname_to_inx = {fname: inx for inx, fname in enumerate(self._img_fnames)}

        self.images_list = None
        if not lazy:
            self.images_list = list(map(self.__getitem__, range(len(self.labels))))

        self.iat_labels = self.gen_labels_pata(iat_type=iat_type)[0]
    
    def _load_pata_sample(self, sample_labels) -> dict:
        res = Dotdict(dict(sample_labels))
        img_fname = res.file
        try:
            res.img = self._transforms(Image.open(img_fname))
        except:
            try:
                res.img = self._transforms(images=Image.open(img_fname))
            except: pass
        return res
    
    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_pata_sample(self.labels.iloc[index])
        ff_sample.iat_label = self.iat_labels[index]
        ff_sample.orig_labels_list = self._img_fnames[index]
        return ff_sample

    def __len__(self):
        return len(self.labels)
    


class ImgNet(IATDataset):
    def __init__(self, lazy: bool = True, mode: str = "validation",
                 _n_samples: Union[float, int] = None, transforms: Callable = None):
        
        ds = load_dataset(
            'frgfm/imagenette',
            '320px',
            split=mode,
            revision="4d512db",
            trust_remote_code=True
        )
        # ds = load_dataset(
        #     'ILSVRC/imagenet-1k',
        #     split=mode,
        #     trust_remote_code=True
        # )
        labels = ds.info.features['label'].names
        self.prompts = [f"a photo of a {label}" for label in labels]
        #['tench',
        # 'English springer',
        # 'cassette player',
        # 'chain saw',
        # 'church',
        # 'French horn',
        # 'garbage truck',
        # 'gas pump',
        # 'golf ball',
        # 'parachute']

        DATA_PATH = (pathlib.Path(__file__) / ".." / ".." / "data").resolve().absolute()
        self.DATA_PATH = DATA_PATH / 'imagenet'

        self._transforms = (lambda x: x) if transforms is None else transforms

        data = []
        # Loop through each emotion folder
        for i in range(len(ds)):
            img = ds[i]['image']
            lab = ds[i]['label']

            # Create a safe filename
            fname = f"{i}_{lab}.png"
            fpath = self.DATA_PATH / fname

            data.append([str(fpath), lab])

            if not fpath.exists():  # avoid redownloading
                img = img.convert("RGB")
                img.save(fpath)
            continue

        # Create a DataFrame
        self.labels = pd.DataFrame(data, columns=['file', 'label'])

        if _n_samples is not None:
            if isinstance(_n_samples, float):
                _n_samples = int(len(self.labels) * _n_samples)

            self.labels = self.labels[:_n_samples]

        self._img_fnames = [x for x in self.labels["file"]]
        self._fname_to_inx = {fname: inx for inx, fname in enumerate(self._img_fnames)}

        self.images_list = None
        if not lazy:
            self.images_list = list(map(self.__getitem__, range(len(self.labels))))

        self.iat_labels = self.gen_labels_imagenet()[0]
    
    def _load_imagenet_sample(self, sample_labels) -> dict:
        res = Dotdict(dict(sample_labels))
        img_fname = res.file
        try:
            res.img = self._transforms(Image.open(img_fname))
        except:
            try:
                res.img = self._transforms(images=Image.open(img_fname))
            except: pass
        return res
    
    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_imagenet_sample(self.labels.iloc[index])
        ff_sample.iat_label = self.iat_labels[index]
        ff_sample.orig_labels_list = self._img_fnames[index]
        return ff_sample

    def __len__(self):
        return len(self.labels)



class CocoGender(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)

    def __getitem__(self, index):
        # Get image and label from parent class
        img, label = super().__getitem__(index)
        
        # You can customize 'iat_label' if needed (currently same as label)
        return {
            "img": img,
            "label": label,
            "iat_label": label
        }


class VLA(Dataset):
    def __init__(self, root, transform=None, seed=42):
        self.root = root
        self.transform = transform

        random.seed(seed)
        male_samples = []
        female_samples = []

        # iterate over 5 subfolders
        for subdir in ['fairface_margin025', 'fairface_margin125', 'miap', 'pata', 'phase']: #sorted(os.listdir(root)):
            sub_path = os.path.join(root, subdir)

            if not os.path.isdir(sub_path):
                continue

            img_dir = os.path.join(sub_path, "images")
            anno_path = os.path.join(sub_path, "annotations.csv")

            if not os.path.exists(img_dir) or not os.path.exists(anno_path):
                continue

            with open(anno_path, newline='') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    img_name = row["name"]
                    gender = row["gender"]

                    img_path = os.path.join(img_dir, img_name)
                    if not os.path.exists(img_path):
                        continue

                    try:
                        with Image.open(img_path) as img:
                            img.verify()   # 파일 무결성 검사 (메모리 로드 안 함)
                    except Exception:
                        print(f"[INIT WARNING] Corrupted image removed: {img_path}")
                        continue

                    # normalize gender → 0/1
                    if str(gender).lower() in ["male", "man"]:
                        label = 0
                        male_samples.append((img_path, label))
                    else:
                        label = 1
                        female_samples.append((img_path, label))
        
        min_count = min(len(male_samples), len(female_samples))

        male_samples = random.sample(male_samples, min_count)
        female_samples = random.sample(female_samples, min_count)

        self.samples = male_samples + female_samples

        total = len(self.samples)

        print(f"[VLA] Balanced dataset loaded with {total} samples")
        print(f"  Male   : {min_count} (50.00%)")
        print(f"  Female : {min_count} (50.00%)")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return {
            "img": img,
            "label": label,
            "iat_label": label
        }



class Celeba(ImageFolder):
    def __init__(self, root, split='train', transform=None):
        super().__init__(root=root, transform=transform)
        self.img_root = os.path.join(root, 'img_align_celeba')
        self.label_path = os.path.join(root, 'list_attr_celeba.txt')

        with open(self.label_path, 'r') as f:
            lines = f.readlines()

        header = lines[0].strip().split()
        male_idx = header.index("Male")  # column index of 'Male' attribute

        self.samples = []
        male_count = 0
        female_count = 0
        for line in lines[2:]:
            parts = line.strip().split()
            img_name = parts[0]
            attrs = [int(x) for x in parts[1:]]
            
            # Male: 1 → 1, -1 → 0
            male_label = 1 if attrs[male_idx] == 1 else 0
            if male_label == 1:
                male_count += 1
            else:
                female_count += 1
            
            self.samples.append((img_name, male_label))
        
        total = male_count + female_count
        print(f"[Celeba] Dataset loaded with {total} samples")
        print(f"  Male   : {male_count} ({male_count/total:.2%})")
        print(f"  Female : {female_count} ({female_count/total:.2%})")

    
    def __getitem__(self, index):
        img_name, label = self.samples[index]
        img_path = os.path.join(self.img_root, img_name)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return {
            "img": img,
            "label": label,
            "iat_label": label  # if you want a separate key for compatibility
        }

class SBBench(Dataset):
    def __init__(self, d_type, attribute_type, is_crop, transform):
        
        if attribute_type == 'age':
            attribute_name = 'age' #'old'
        elif attribute_type == 'gender':
            attribute_name = 'gender' #'male'
        
        if is_crop:
            crop_name = 'True'
        else:
            crop_name = 'False'

        self.transform = transform
        self.ds = load_dataset(f"vlmbias/sbbench_{d_type}_{attribute_name}_crop_{crop_name}")['train']
       

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]
        img = example["image"]
        img = self.transform(img)
        label = example["label"]
        return {"img": img, "label": label, "iat_label": label}


class CocoGenderTxtDataset(Dataset):
    def __init__(self, root, split="train"):
        self.df = pd.read_csv(os.path.join(root, f'{split}.csv'))
        self.captions = self.df['caption'].values
        self.genders = self.df['gender'].values
        self.iat_label_dic = {'Male': 0, 'Female': 1}
        self.iat_labels = [self.iat_label_dic[gender_name] for gender_name in self.genders]

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