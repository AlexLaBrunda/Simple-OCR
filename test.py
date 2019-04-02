from train import dotrianing, findfeatures_training
from scipy.spatial.distance import cdist
from skimage import io
import matplotlib.pyplot as plt


trainedFeatures = dotrianing()
testFeatures = findfeatures_training('test1.bmp', False)
D = cdist(testFeatures, trainedFeatures)
print D
io.imshow(D)
plt.title('Distance Matrix')
io.show()
