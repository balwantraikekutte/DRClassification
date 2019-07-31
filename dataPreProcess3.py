import numpy as np

import cv2

from keras.utils import np_utils

immatrix3 = np.load('immatrix3.npy')
imlabel3 = np.load('imlabel3.npy')

# Local average colour subtraction
scale = 300
immatrix33 = []

for i in range(0,(len(immatrix3))):
    a = np.zeros((512,512,3))
    a = immatrix3[i,:,:,:]
    b = np.zeros(a.shape)
    cv2.circle(b,(int(a.shape[1]/2),int(a.shape[0]/2)),int(scale*0.9),(1,1,1),-1,8,0)
    aa=cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale/30),-4,128)*b+128*(1-b)
    immatrix33.append(aa)

immatrix33 = np.asarray(immatrix33)  

# Convert variables to categorical
nb_classes = 5

imlabel3 = np_utils.to_categorical(imlabel3, nb_classes)


# Save
np.save('immatrix3.npy', immatrix33)
np.save('imlabel3.npy', imlabel3)
