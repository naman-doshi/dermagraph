from train import get_model, path
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
from flask import Flask, jsonify
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from flask import Flask, render_template, request, redirect, url_for
import urllib.request
import base64
from io import BytesIO
import json
from flask import after_this_request, make_response
import matplotlib.pyplot as plt
from torchvision.utils import save_image


device = torch.device('mps')
app = Flask(__name__)
solved = False

class MelanomaDataset(Dataset):
    def __init__(self, df, img):
        self.df = df
        self.img = img
        
    def __getitem__(self, indx):
        return self.img
    
    def __len__(self):
        return self.df.shape[0]
    

def render(text):
    return render_template('index.html', result=text)

@app.route('/', methods = ['GET'])
def index(): 
    return render_template(("index.html"))

@app.route('/predict/input', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        file = request.form
        params = file['params']
        params = json.loads(params)
        photo = str(file['photo'])
        photo = urllib.request.urlopen(photo).read()
        photo = base64.b64encode(photo)
        embed = get_prediction(params, photo)
        return str(embed['prob'])

def transform_image(img_bytes):
  p = 0.5
  imagenet_stats = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
  image = Image.open(BytesIO(base64.b64decode(img_bytes)))
  width, height, = image.size
  if width < height:
    sm = width
  else:
    sm = height
  transforms = A.Compose([
        A.CenterCrop(height=sm, width=sm),
        A.Resize(height=256, width=256),
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

  
  return transforms(**{"image": np.array(image)})["image"]

def get_prediction(params, img_bytes):
  tensor = transform_image(img_bytes)
  model, opt = get_model(model_name='efficientnet-b0', lr=1e-4, wd=1e-4)
  model.load_state_dict(torch.load(f'src/effb0.pth', map_location=device))
  model.eval()

  df = pd.DataFrame(columns=['image_name', 'patient_id', 'sex', 'age_approx', 'anatom_site_general_challenge', 'width', 'height'])
  entry = pd.DataFrame.from_dict(params)
  test_df = pd.concat([df, entry], ignore_index=True)
  test_ds = MelanomaDataset(df=test_df, img=tensor)
  test_dl = DataLoader(dataset=test_ds, batch_size=1, shuffle=False, num_workers=4)

  tta = 10
  preds = np.zeros(len(test_ds))
  for tta_id in range(tta):
      test_preds = []
      with torch.no_grad():
          for xb in test_dl:
              xb = xb.to(device)
              out = model.to(device)(xb)
              out = torch.sigmoid(out)
              test_preds.extend(out.cpu().detach().numpy())
          preds += np.array(test_preds).reshape(-1)
  preds /= tta
  preds = {"prob": preds.tolist()[0]}
  return preds

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)