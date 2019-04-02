import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
from skimage.morphology import closing, square, dilation
from skimage.filters import threshold_otsu
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle


letters = []
pkl_file = open('test1_gt.pkl', 'rb')
mydict = pickle.load(pkl_file)
pkl_file.close()
classes = mydict['classes']
locations = mydict['locations']


def findfeatures_training(filename, ploton):
    # type: (basestring, bool) -> list
    img = io.imread(filename)
    fileletter = filename[0]
    if ploton:
        print img.shape
        io.imshow(img)
        plt.title('Original Image')
        io.show()
        hist = exposure.histogram(img)
        plt.bar(hist[1], hist[0])
        plt.title('Histogram')
        plt.show()
    th = threshold_otsu(img)
    img_binary = (img < th).astype(np.double)
    if ploton:
        print th
        io.imshow(img_binary)
        plt.title('Binary Image')
        io.show()
    selem = square(1)
    dilated_img_binary = dilation(img_binary, selem)
    if ploton:
        io.imshow(dilated_img_binary)
        plt.title('closed')
        io.show()
    img_label = label(dilated_img_binary, background=0)
    if ploton:
        io.imshow(img_label)
        plt.title('Labeled Image')
        io.show()
        print np.amax(img_label)
    regions = regionprops(img_label)
    ax = plt.gca()
    if ploton:
        io.imshow(dilated_img_binary)
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
            letters.append(fileletter)
    if ploton:
        print count
        io.show()
    return features


def column(matrix, i):
    return [row[i] for row in matrix]


def dotrianing():
    features = findfeatures_training('a.bmp', False)
    features.extend(findfeatures_training('d.bmp', False))
    features.extend(findfeatures_training('m.bmp', False))
    features.extend(findfeatures_training('n.bmp', False))
    features.extend(findfeatures_training('o.bmp', False))  # problem letter
    features.extend(findfeatures_training('p.bmp', False))
    features.extend(findfeatures_training('q.bmp', False))
    features.extend(findfeatures_training('r.bmp', False))
    features.extend(findfeatures_training('u.bmp', False))  # problem letter
    features.extend(findfeatures_training('w.bmp', False))
    return StandardScaler(True, False, False).fit_transform(features)


def doTrainingWithConfusion():
    f2 = dotrianing()
    D = cdist(f2, f2)
    print D
    io.imshow(D)
    plt.title('Distance Matrix')
    io.show()
    D_index = np.argsort(D, axis=1)
    print D_index
    indexofbestguess = column(D_index, 1)
    labelofbestguess = []
    for x in indexofbestguess:
        labelofbestguess.append(letters[x])
    confM = confusion_matrix(letters, labelofbestguess, labels=["a", "d", "m", "n", "o", "p", "q", "r", "u", "w"])
    print confM
    io.imshow(confM)
    plt.title('Confusion Matrix')
    io.show()


def find_ground_truth(t):
    index = 0
    for (x, y) in locations:
        minr, minc, maxr, maxc = t.bbox
        if x < maxc and x > minc and y < maxr and y > minr:
            return classes[index]
        index = index + 1


# def find_ground_truth(t):
    # [tx, ty] = t.centroid
    # loc_dist = cdist([[tx, ty]], locations)
    # closest_ctr_idx = np.argmin(loc_dist, axis=1)
    # return classes[closest_ctr_idx]


def findfeatures_testing(filename, ploton):
    # type: (basestring, bool) -> list
    img = io.imread(filename)
    fileletter = filename[0]
    if ploton:
        print img.shape
        io.imshow(img)
        plt.title('Original Image')
        io.show()
        hist = exposure.histogram(img)
        plt.bar(hist[1], hist[0])
        plt.title('Histogram')
        plt.show()
    th = threshold_otsu(img)
    img_binary = (img < th).astype(np.double)
    if ploton:
        print th
        io.imshow(img_binary)
        plt.title('Binary Image')
        io.show()
    selem = square(1)
    dilated_img_binary = dilation(img_binary, selem)
    if ploton:
        io.imshow(dilated_img_binary)
        plt.title('closed')
        io.show()
    img_label = label(dilated_img_binary, background=0)
    if ploton:
        io.imshow(img_label)
        plt.title('Labeled Image')
        io.show()
        print np.amax(img_label)
    regions = regionprops(img_label)
    ax = plt.gca()
    if ploton:
        io.imshow(dilated_img_binary)
        plt.title('Bounding boxes')
    features = []
    tt = []
    count = 0
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        if maxc - minc > 9 and maxr - minr > 9:
            tt.append(find_ground_truth(props))
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
            letters.append(fileletter)
    if ploton:
        print count
        io.show()
    return features, tt

