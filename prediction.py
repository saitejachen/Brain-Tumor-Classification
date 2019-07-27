import pydicom
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense , Activation, Dropout, Flatten,Conv2D, MaxPooling2D,Conv1D,MaxPooling1D
from keras.models import load_model

img=np.zeros(shape=(256,256,3))
I=pydicom.dcmread('000005.dcm').pixel_array
img[:,:,0]=I*255/(np.amax(I))
img[:,:,1]=img[:,:,0]
img[:,:,2]=img[:,:,0]

plt.imshow(img)
plt.show()
print(img.shape)

img=np.expand_dims(img,axis=2)
img=np.swapaxes(img,0,2)
img=np.swapaxes(img,1,2)
model=load_model('cnn.h5')
prediction=model.predict(img)
print(prediction)

if prediction == [[1]]:
	print('Tumor')
else:
	print('No Tumor')