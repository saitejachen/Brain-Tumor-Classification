import numpy as np
import pydicom
import math
from skimage.restoration import denoise_nl_means, estimate_sigma
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)
def store_evolution_in(lst):

    def _store(x):
        lst.append(np.copy(x))

    return _store
I = pydicom.dcmread('000001.dcm')
I=I.pixel_array
x=I.copy()
y=x.resize((256,256))
max=np.amax(I)
sigma = 0.08
sigma_est = np.mean(estimate_sigma(I, multichannel=False))
patch_kw = dict(patch_size=5,     
                patch_distance=6,  
                multichannel=False)
I = denoise_nl_means(I, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)
image =np.copy(I)
print(image[10,10])
print(image[100,150])
print(np.amax(image[50,190:210]))
print(np.amax(image))
max=np.amax(image)
eimage=(image*255/max)**5
#eimage = image ** (3)    
init_ls = checkerboard_level_set(image.shape, 6)
evolution = []
callback = store_evolution_in(evolution)
ls = morphological_chan_vese(eimage, 35, init_level_set=init_ls, smoothing=3,
                             iter_callback=callback)

plt.imshow(image,cmap=plt.cm.bone)
plt.contour(ls, [0.5], colors='r')
plt.show()