from keras.layers import Dense , Activation, Dropout, Flatten,Conv2D, MaxPooling2D,Conv1D,MaxPooling1D
from keras.models import Sequential
from skimage.restoration import denoise_nl_means, estimate_sigma
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.utils import shuffle


def denoise(I):
	sigma = 0.08
	sigma_est = np.mean(estimate_sigma(I, multichannel=True))
	patch_kw = dict(patch_size=5,     
                patch_distance=6,  
                multichannel=True)
	I = denoise_nl_means(I, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)
	return I


i=0
path = 'C:\\Users\\saite\\Desktop\\T&T\\Training Data\\new tumor./*.dcm'	
files = glob.glob(path)
print(len(files))
I=np.zeros(shape=(256,256,3,176))
for name in files:
	print(files[i])
	img=pydicom.dcmread(files[i]).pixel_array
	I[:,:,0,i]=img*255/(np.amax(img))
	I[:,:,1,i]=I[:,:,0,i]
	I[:,:,2,i]=I[:,:,0,i]
	#I[:,:,:,i]=denoise(I[:,:,:,i])
	i=i+1

k=0
path = 'C:\\Users\\saite\\Desktop\\T&T\\Training Data\\new normal./*.dcm'	
files = glob.glob(path)
print(len(files))
for name in files:
	print(files[k])
	img=pydicom.dcmread(files[k]).pixel_array
	I[:,:,0,i]=img*255/(np.amax(img))
	I[:,:,1,i]=I[:,:,0,i]
	I[:,:,2,i]=I[:,:,0,i]
	#I[:,:,:,i]=denoise(I[:,:,:,i])
	i=i+1
	k=k+1

f = open('I.pckl', 'wb')
pickle.dump(I, f)
f.close()



f = open('I.pckl', 'rb')
I = pickle.load(f)
f.close()
o_t=np.ones(shape=(1,88))
o_n=np.zeros(shape=(1,88))
#I=np.concatenate([I_t,I_n],2)
o = np.append(o_t,o_n)
print(I.shape)
print(o.shape)
'''
randomize = np.arange(len(I[1,1,:]))
np.random.shuffle(randomize)
I[:,:,] = I[:,:,	randomize]
o = o[randomize]
print(I.shape)
'''
'''
I=np.expand_dims(I,axis=3)
I=np.swapaxes(I,0,3)
I=np.swapaxes(I,1,3)
I=np.swapaxes(I,2,3)
'''
I=np.swapaxes(I,0,3)
I=np.swapaxes(I,2,3)


model=Sequential()
model.add(Conv2D(256,(3,3), input_shape=(256,256,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3), input_shape=(256,256,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3), input_shape=(256,256,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

#model.fit(I, o,epochs=8,batch_size=11)
model.fit(I, o,epochs=44,batch_size=4)
score = model.evaluate(I, o, batch_size=4	)
#model.fit_generator(I, samples_per_epoch=8,nb_epoch=11,validation_data=o, nb_val_samples=176)
model.save('cnn.h5')

