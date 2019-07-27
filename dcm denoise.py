import numpy as np
import matplotlib.pyplot as plt
import pydicom
from skimage import color
from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.measure import compare_psnr

noisy = pydicom.dcmread('000017.dcm')
noisy=noisy.pixel_array

sigma = 0.08



sigma_est = np.mean(estimate_sigma(noisy, multichannel=False))


patch_kw = dict(patch_size=5,     
                patch_distance=6,  
                multichannel=False)


denoise = denoise_nl_means(noisy, h=2.15 * sigma_est, fast_mode=False,
                           **patch_kw)



fig , ax =plt.subplots(nrows=1,ncols=2)

ax[0].imshow(noisy)
ax[1].imshow(denoise)


plt.show()

