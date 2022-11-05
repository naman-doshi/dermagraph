# By Team DermaGraph, 2022
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from fastprogress.fastprogress import master_bar, progress_bar
from sklearn.metrics import roc_auc_score
from efficientnet_pytorch import EfficientNet
import albumentations as A
from albumentations.pytorch.transforms import ToTensor

# mps for Apple Silicon, gpu for a Nvidia GPU, or cpu otherwise.
device = torch.device("cpu")

def list_files(path:Path):
    return [o for o in path.iterdir()]

path = Path('input')

# Defining the MelanomaDataset Class.
class MelanomaDataset(Dataset):
    def __init__(self, df, img_path_one, transforms=None, is_test=False):
        self.df = df
        self.img_path_one = img_path_one
        self.transforms = transforms
        self.is_test = is_test
        
    def __getitem__(self, indx):
        img_path = f"{self.img_path_one}/{self.df.iloc[indx]['image_name']}.jpg"
        img = Image.open(img_path)
 
        if self.transforms:
            img = self.transforms(**{"image": np.array(img)})["image"]
            
        if self.is_test:
            return img

        target = self.df.iloc[indx]['target']
        return img, torch.tensor([target], dtype=torch.float32)
    
    def __len__(self):
        return self.df.shape[0]

# Defining the EffientNet Model and its parameters.
class Model(nn.Module):
    def __init__(self, model_name='efficientnet-b0', pool=F.adaptive_avg_pool2d):
        super().__init__()
        self.pool = pool
        self.backbone = EfficientNet.from_pretrained(model_name)
        in_features = getattr(self.backbone,'_fc').in_features
        self.classifier = nn.Linear(in_features,1)

    def forward(self, x):
        features = self.pool(self.backbone.extract_features(x),1)
        features = features.view(x.size(0),-1)
        return self.classifier(features)

# Function to initialise the model that it used in server.py
def get_model(model_name='efficientnet-b0', lr=1e-5, wd=0.01, freeze_backbone=False, opt_fn=torch.optim.AdamW, device=None):
    model = Model(model_name=model_name)

    # Freezing Layers
    if freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    
    # Optimising weights
    opt = opt_fn(model.parameters(), lr=lr, weight_decay=wd)
    model = model.to(device)
    return model, opt

      