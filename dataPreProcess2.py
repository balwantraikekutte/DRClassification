import numpy as np

import cv2

from keras.utils import np_utils

immatrix2 = np.load('immatrix2.npy')
imlabel2 = np.load('imlabel2.npy')

# Local average colour subtraction
scale = 300
immatrix22 = []

for i in range(0,(len(immatrix2))):
    a = np.zeros((512,512,3))
    a = immatrix2[i,:,:,:]
    b = np.zeros(a.shape)
    cv2.circle(b,(int(a.shape[1]/2),int(a.shape[0]/2)),int(scale*0.9),(1,1,1),-1,8,0)
    aa=cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale/30),-4,128)*b+128*(1-b)
    immatrix22.append(aa)
    
immatrix22 = np.asarray(immatrix22)

# Convert variables to categorical
nb_classes = 5

imlabel2 = np_utils.to_categorical(imlabel2, nb_classes)


# Save
np.save('immatrix2.npy', immatrix22)
np.save('imlabel2.npy', imlabel2)
