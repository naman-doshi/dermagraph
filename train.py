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
device = torch.device("mps")

def list_files(path:Path):
    return [o for o in path.iterdir()]

path = Path('input')

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

def get_augmentations(p=0.5):

    # Assigning the ImageNet mean and standard deviation to a variable
    imagenet_stats = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}

    # Using albumentations' augmentation pipeline to sequentially alter the images.
    # The variable p stands for the percentage chance of a particular alteration happening.
    train_tf = A.Compose([
        A.Cutout(p=p),
        A.RandomRotate90(p=p),
        A.Flip(p=p),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2,
                                       ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=50,
                val_shift_limit=50)
        ], p=p),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=p),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=p),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1)
        ], p=p), 
        ToTensor(normalize=imagenet_stats)
        ])
    
    
    test_tf = A.Compose([
        A.Cutout(p=p),
        A.RandomRotate90(p=p),
        A.Flip(p=p),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2,
                                       ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=50,
                val_shift_limit=50)
        ], p=p),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=p),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=p),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1)
        ], p=p), 
        ToTensor(normalize=imagenet_stats)
        ])

    
    return train_tf, test_tf

def get_train_val_split(df):
    # Removal of duplicates
    df = df[df.tfrecord != -1].reset_index(drop=True)

    # Splitting
    train_tf_records = list(range(len(df.tfrecord.unique())))[:12]
    split_cond = df.tfrecord.apply(lambda x: x in train_tf_records)
    train = df[split_cond].reset_index()
    valid = df[~split_cond].reset_index()
    return train, valid

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

# Training
def training(xb, yb, model, loss_fn, opt, device, scheduler):
    xb, yb = xb.to(device), yb.to(device)
    out = model(xb)
    opt.zero_grad()
    loss = loss_fn(out,yb)
    loss.backward()
    opt.step()
    scheduler.step()
    return loss.item()

# Validation
def validation(xb,yb,model,loss_fn,device):
    xb,yb = xb.to(device), yb.to(device)
    out = model(xb)
    loss = loss_fn(out,yb)
    out = torch.sigmoid(out)
    return loss.item(),out

def get_data(train_df, valid_df, train_tfms, test_tfms, bs):
    train_ds = MelanomaDataset(df=train_df, img_path_one=path/'train', transforms=train_tfms)
    valid_ds = MelanomaDataset(df=valid_df, img_path_one=path/'train', transforms=test_tfms)
    train_dl = DataLoader(dataset=train_ds, batch_size=bs, shuffle=True, num_workers=4)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=bs*2, shuffle=False, num_workers=4)
    return train_dl, valid_dl

def fit(epochs, model, train_dl, valid_dl, opt, devic=None, loss_fn=F.binary_cross_entropy_with_logits):
    devic = device
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, len(train_dl)*epochs)
    val_rocs = [] 
    
    #Creating progress bar
    mb = master_bar(range(epochs))
    mb.write(['epoch','train_loss','valid_loss','val_roc'],table=True)

    for epoch in mb:    
        trn_loss,val_loss = 0.0,0.0
        val_preds = np.zeros((len(valid_dl.dataset),1))
        val_targs = np.zeros((len(valid_dl.dataset),1))
        
        #Training
        model.train()
        
        #For every batch 
        for xb,yb in progress_bar(train_dl,parent=mb):
            trn_loss += training(xb.to(device), yb.to(device), model.to(device), loss_fn, opt, devic, scheduler) 
        trn_loss /= mb.child.total

        #Validation
        model.eval()
        with torch.no_grad():
            for i,(xb,yb) in enumerate(progress_bar(valid_dl, parent=mb)):
                loss,out = validation(xb, yb, model, loss_fn, devic)
                val_loss += loss
                bs = xb.shape[0]
                val_preds[i*bs:i*bs+bs] = out.cpu().numpy()
                val_targs[i*bs:i*bs+bs] = yb.cpu().numpy()

        val_loss /= mb.child.total
        val_roc = roc_auc_score(val_targs.reshape(-1),val_preds.reshape(-1))
        val_rocs.append(val_roc)

        loss = round(trn_loss, 6)
        mb.write([str(epoch), str(round(trn_loss, 6)), str(round(val_loss, 6)), str(round(float(val_roc), 6))], table=True)
    return model,val_rocs

if __name__ ==  '__main__':
    df = pd.read_csv(path/'train.csv')
    train_df, valid_df = get_train_val_split(df)
    train_tfms, test_tfms = get_augmentations(p=0.5)
    train_dl, valid_dl = get_data(train_df, valid_df, train_tfms, test_tfms, 16)
    model, opt = get_model(model_name='efficientnet-b1', lr=1e-4, wd=1e-4)
    model, val_rocs = fit(8, model, train_dl, valid_dl, opt)
    print(val_rocs)
    torch.save(model.state_dict(), f'effb1.pth')

    model, opt = get_model(model_name='efficientnet-b1',lr=1e-4,wd=1e-4)
    model.load_state_dict(torch.load(f'effb1.pth', map_location=device))
    model.eval()
    test_df = pd.read_csv(path/'test.csv')
    test_ds = MelanomaDataset(df=test_df, img_path_one=path/'test',transforms=test_tfms, is_test=True)
    test_dl = DataLoader(dataset=test_ds, batch_size=32, shuffle=False,num_workers=4)
    
    tta = 1
    preds = np.zeros(len(test_ds))
    for tta_id in range(tta):
        count = 0
        test_preds = []
        with torch.no_grad():
            for xb in test_dl:
                print(count)
                xb = xb.to(device)
                out = model.to(device)(xb)
                out = torch.sigmoid(out)
                test_preds.extend(out.cpu().detach().numpy())
                print(test_preds)
                count += 1
            preds += np.array(test_preds).reshape(-1)
        print(f'TTA {tta_id+1}')
    preds /= tta

    subm = pd.read_csv(path/'sample_submission.csv')
    subm.target = preds
    subm.to_csv('submissions/submission2.csv', index=False)

      