import os

import numpy as np
import pandas as pd

from PIL import Image

trainLabels = pd.read_csv("/local/data/chaitanya/DR/classification/dr_class_seperated_data/trainingLabels.csv")

listing = os.listdir("/local/data/chaitanya/DR/classification/dr_class_seperated_data/trainData3") 

img_rows, img_cols = 512, 512

immatrix = []
imlabel = []

for file in listing:
    base = os.path.basename("/local/data/chaitanya/DR/classification/dr_class_seperated_data/trainData3/" + file)
    fileName = os.path.splitext(base)[0]
    imlabel.append(trainLabels.loc[trainLabels.image==fileName, 'level'].values[0])
    im = Image.open("/local/data/chaitanya/DR/classification/dr_class_seperated_data/trainData3/" + file)
    img = im.resize((img_rows,img_cols))
    img = np.asarray(img)
    img = img.astype(np.float64)
    immatrix.append(img)  

immatrix = np.asarray(immatrix)
imlabel = np.asarray(imlabel)

np.save('immatrix3.npy', immatrix)
np.save('imlabel3.npy', imlabel)
