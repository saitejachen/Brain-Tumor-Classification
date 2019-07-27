import numpy as np
import medpy
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import color
from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.measure import compare_psnr


img=nib.load('t3.mnc')
d=img.get_data()
img=d[::,60]/np.amax(d)

s = np.linspace(0, 2*np.pi, 400)
x = 130 + 80*np.cos(s)
y = 70 + 80*np.sin(s)
init = np.array([x, y]).T

C=np.zeros(shape=(256,256))
C[0,:]=1
C[255,:]=1
C[:,0]=1
C[:,255]=1


img = pydicom.dcmread('000005.dcm')
img=img.pixel_array
img=img*1/565

print(np.amax(img))
sigma = 0.08



sigma_est = np.mean(estimate_sigma(img, multichannel=False))


patch_kw = dict(patch_size=5,     
                patch_distance=6,  
                multichannel=False)


img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)



#img=denoise_bilateral(img, sigma_spatial=15 ,multichannel=False)

snake = active_contour(img,
                       C, alpha=0.015, beta=10, gamma=0.001)
#plt.imshow(C)

plt.imshow(img)
plt.plot(C[:, 0], C[:, 1], '--r', lw=3)
plt.plot(snake[:, 0], snake[:, 1], '-b', lw=3)

plt.axis([0, img.shape[1], img.shape[0], 0])


plt.show()