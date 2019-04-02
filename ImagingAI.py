import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
from skimage.morphology import closing, square, dilation
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle

img = io.imread('a.bmp')
print img.shape
io.imshow(img)
plt.title('Original Image')
io.show()
hist = exposure.histogram(img)
plt.bar(hist[1], hist[0])
plt.title('Histogram')
plt.show()
th = threshold_otsu(img)
print th
img_binary = (img < th).astype(np.double)
io.imshow(img_binary)
plt.title('Binary Image')
io.show()
selem = square(1)
dilated_img_binary = dilation(img_binary, selem)
io.imshow(dilated_img_binary)
plt.title('closed')
io.show()
img_label = label(dilated_img_binary, background=0)
io.imshow(img_label)
plt.title('Labeled Image')
io.show()
print np.amax(img_label)
regions = regionprops(img_label)
io.imshow(dilated_img_binary)
ax = plt.gca()
plt.title('Bounding boxes')
features = []
count = 0
for props in regions:
    minr, minc, maxr, maxc = props.bbox
    if maxc - minc > 9 and maxr - minr > 9:
        count += 1
        ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
        roi = dilated_img_binary[minr:maxr, minc:maxc]
        m = moments(roi)
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]
        mu = moments_central(roi, cr, cc)
        nu = moments_normalized(mu)
        hu = moments_hu(nu)
        features.append(hu)
print count
io.show()
