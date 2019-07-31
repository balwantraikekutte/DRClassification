import numpy as np

import cv2

from keras.utils import np_utils

immatrix1 = np.load('immatrix1.npy')
imlabel1 = np.load('imlabel1.npy')

# Local average colour subtraction
scale = 300
immatrix11 = []

for i in range(0,(len(immatrix1))):
    a = np.zeros((512,512,3))
    a = immatrix1[i,:,:,:]
    b = np.zeros(a.shape)
    cv2.circle(b,(int(a.shape[1]/2),int(a.shape[0]/2)),int(scale*0.9),(1,1,1),-1,8,0)
    aa=cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale/30),-4,128)*b+128*(1-b)
    immatrix11.append(aa)

immatrix11 = np.asarray(immatrix11)

# Convert variables to categorical
nb_classes = 5

imlabel1 = np_utils.to_categorical(imlabel1, nb_classes)


# Save
np.save('immatrix1.npy', immatrix11)
np.save('imlabel1.npy', imlabel1)
