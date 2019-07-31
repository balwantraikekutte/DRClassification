import os, glob
import cv2
import numpy as np
import pandas as pd

def scaleRadius(img,scale):
    x=img[int(img.shape[0]/2),:,:].sum(1)
    r=(x>x.mean()/10).sum()/2
    s=scale*1.0/r
    return cv2.resize(img,(0,0),fx=s,fy=s)

scale = 300
image_length = 512
image_height = 512
num_channels = 3
i = 0

data_file = glob.glob('/local/data/chaitanya/DR/classification/dr_class_seperated_data/trainData1/*.jpeg')

trainLabels = pd.read_csv("/local/data/chaitanya/DR/classification/dr_class_seperated_data/trainingLabels.csv")

trainData = np.zeros((len(data_file),image_length, image_height, num_channels))
trainLabel = []

for f in (data_file):
    a=cv2.imread(f)
    a=scaleRadius(a,scale)
    b=np.zeros(a.shape)
    b=b.astype(np.uint8) 
    cv2.circle(b,(int(a.shape[1]/2),int(a.shape[0]/2)),int(scale*0.9),(1,1,1),-1,8,0)
    aa=cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale/30),-4,128)*b+128*(1-b)
    resized_image = cv2.resize(aa, (image_length, image_height))
    resized_image = resized_image.astype(np.float64)
    trainData[i,:,:,:] = resized_image[:,:,:]
    base = os.path.basename("/local/data/chaitanya/DR/classification/dr_class_seperated_data/trainData1/" + f)
    fileName = os.path.splitext(base)[0]
    trainLabel.append(trainLabels.loc[trainLabels.image==fileName, 'level'].values[0])
    i += 1
    
np.save('trainData1.npy',trainData)
np.save('trainLabel1.npy', trainLabel)
