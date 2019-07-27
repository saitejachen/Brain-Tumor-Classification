from skimage.filters import sobel
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from scipy import ndimage as ndi
from PIL import Image
import cv2
from skimage import feature
from skimage import color
from skimage import io
from skimage.restoration import denoise_nl_means, estimate_sigma
I=cv2.imread('mine turtle.jpg', 0)

I=np.array(I,dtype=float)
I = pydicom.dcmread('000006.dcm')
I=I.pixel_array
max=np.amax(I)
#	I=I*255/max
sigma = 0.08
sigma_est = np.mean(estimate_sigma(I, multichannel=False))
patch_kw = dict(patch_size=5,     
                patch_distance=6,  
                multichannel=False)
I = denoise_nl_means(I, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)

sobel = sobel(I)

smax=np.amax(sobel)

for i in range(1,255):
	for j in range(1,255):
		if sobel[i,j]<	0.3	*smax	:
			sobel[i,j]=0




plt.imshow(I)
plt.show()