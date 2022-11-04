# By Team DermaGraph, 2022
from train import get_model
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from PIL import Image
from flask import Flask, jsonify, request, render_template
import urllib.request
import base64
from io import BytesIO
import json

# Function to make a custom diagnosis & recommendations based on the chance of melanoma.
def getDiagnosis(pred, status):
    if pred <= 3:
        diagnosis = "Safe 🎉🎉"
        if status=='patient':
            recs = f"DermaGraph believes that this lesion is safe. The chance of this lesion being final-stage malignant is {pred}% — which is extremely low. If you do not have the means to do so, visiting a dermatologist is not needed. Of course, please exercise your own judgement, and we encourage you to visit your dermatologist if you feel the need to. Still, if you are experiencing any symptoms — including the lesion becoming raised or irregular, or changes to your lymph nodes and weight — please get it checked immediately."
        else:
            recs = f'DermaGraph believes that this lesion is safe. The chance of this lesion being final-stage malignant is {pred}% — which is extremely low. A biopsy is likely not needed, but please exercise your own judgement.'
    elif pred <= 15:
        diagnosis = "Probably Safe 🎉"
        if status=='patient':
            recs = f"DermaGraph believes that this lesion is probably safe. The chance of this lesion being final-stage malignant is low ({pred}%), but higher than normal — meaning you must pay close attention to this lesion in the future. If you do not have the means to do so, visiting a dermatologist is not needed. Still, if you are experiencing any symptoms — including the lesion becoming raised or irregular, or changes to your lymph nodes and weight — please get it checked immediately."
        else:
            recs = f"DermaGraph believes that this lesion is probably safe. The chance of this lesion being final-stage malignant is low ({pred}%), but higher than normal. A biopsy is likely not needed, but please exercise your own judgement. Please ask the patient to pay close attention to this lesion and to immediately inform you if they experience any other symptoms."
    elif pred <= 35:
        diagnosis = "Probably Early-Stage Malignant 🚨"
        if status=='patient':
            recs = f"DermaGraph believes that this lesion is early-stage malignant, with a {pred}% chance of final-stage malignancy. Please visit your dermatologist immediately. If you experience any symptoms — including the lesion becoming raised or irregular, or changes to your lymph nodes and weight — please get it checked again."
        else:
            recs = f"DermaGraph believes that this lesion is early-stage malignant, with a {pred}% chance of final-stage malignancy. Please conduct a biopsy and ask the patient to immediately inform you if they experience any other symptoms."
    elif pred <= 50:
        diagnosis = "Probably Malignant 🚨🚨"
        if status=='patient':
            recs = f"DermaGraph believes that this lesion might be in the middle stages of cancer, with a {pred}% chance of final-stage malignancy — it is not too late to get it treated. Please visit your dermatologist immediately. If you experience any symptoms — including the lesion becoming raised or irregular, or changes to your lymph nodes and weight — please get it checked again."
        else:
            recs = f"DermaGraph believes that this lesion might be in the middle stages of cancer, with a {pred}% chance of final-stage malignancy. Please conduct a biopsy and ask the patient to immediately inform you if they experience any other symptoms."
    else:
        diagnosis = "Malignant 🚨🚨🚨"
        if status=='patient':
            recs = f"This lesion is almost certainly severely malignant. Please visit your dermatologist immediately. If you experience any more symptoms — including the lesion becoming raised or irregular, or changes to your lymph nodes and weight — please get it checked again."
        else:
            recs = f"It is almost certain that this lesion is severely malignant. Please conduct a biopsy, operate on it as soon as possible, and ask the patient to immediately inform you if they experience any other symptoms."

    return pred, diagnosis, recs


device = torch.device('cpu')
app = Flask(__name__)

# Initialisation of simplified MelanomaDataset class that is optimised for testing usage.
class MelanomaDataset(Dataset):
    def __init__(self, df, img):
        self.df = df
        self.img = img

    def __getitem__(self, indx):
        return self.img

    def __len__(self):
        return self.df.shape[0]

# Flask route to render index.html at all times when a user is viewing it (via a GET request).
@app.route('/', methods = ['GET'])
def index():
    return render_template(("index.html"))

# Flask route to a special endpoint that is designated only for internal communications
@app.route('/predict/input', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Processing the data from the frontend
        file = request.form
        params = file['params']
        params = json.loads(params)
        status = params['status']
        del params['status']
        params['age_approx'] = [float(params['age_approx'][0])]
        photo = str(file['photo'])
        photo = urllib.request.urlopen(photo).read()
        photo = base64.b64encode(photo)
        image = Image.open(BytesIO(base64.b64decode(photo)))
        width, height, = image.size
        params['width'] = [width]
        params['height'] = [height]
        # Getting the probability of melanoma
        embed = get_prediction(params, photo)
        prob = round(float(embed['prob'])*100, 2)
        prob, diagnosis, recs = getDiagnosis(prob, status)
        return jsonify({'prob':prob, 'diag':diagnosis, 'recs':recs})

def transform_image(img_bytes):
  p = 0.5
  imagenet_stats = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
  image = Image.open(BytesIO(base64.b64decode(img_bytes)))
  width, height, = image.size
  if width < height:
    sm = width
  else:
    sm = height

  # The transformations for the image augmentation — an unconventionally large amount in order to account for all types of pictures.
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


# Function to get the prediction based on parameters and image.
def get_prediction(params, img_bytes):
  # Initialising input and model
  tensor = transform_image(img_bytes)
  model, opt = get_model(model_name='efficientnet-b0', lr=1e-4, wd=1e-4)
  model.load_state_dict(torch.load(f'effb0.pth', map_location=device))
  model.eval()
  
  # Preliminarily processing data.
  df = pd.DataFrame(columns=['image_name', 'patient_id', 'sex', 'age_approx', 'anatom_site_general_challenge', 'width', 'height'])
  entry = pd.DataFrame.from_dict(params)
  test_df = pd.concat([df, entry], ignore_index=True)
  test_ds = MelanomaDataset(df=test_df, img=tensor)
  test_dl = DataLoader(dataset=test_ds, batch_size=1, shuffle=False, num_workers=4)

  # TTA iterations.
  tta = 20
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