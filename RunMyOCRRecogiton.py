from train import dotrianing, findfeatures_testing, findfeatures_training, column, letters
from scipy.spatial.distance import cdist
from skimage import io
import matplotlib.pyplot as plt
import pickle
import numpy as np


pkl_file = open('test1_gt.pkl', 'rb')
mydict = pickle.load(pkl_file)
pkl_file.close()
classes = mydict['classes']
locations = mydict['locations']


trainedFeatures = dotrianing()
testFilename = raw_input('Enter the name/path of the test file: ')
trainingImage = False
fileLetter = testFilename[0]
if testFilename == 'a.bmp' or testFilename == 'd.bmp' or testFilename == 'm.bmp' or testFilename == 'n.bmp' or\
        testFilename == 'o.bmp' or testFilename == 'p.bmp' or testFilename == 'q.bmp' or testFilename == 'r.bmp' or\
        testFilename == 'u.bmp' or testFilename == 'w.bmp':
    trainingImage = True
    testFeatures = findfeatures_training(testFilename, False)
if not trainingImage:
    testFeatures, truetestlabels = findfeatures_testing(testFilename, False)
    print truetestlabels
D = cdist(testFeatures, trainedFeatures)
print D
io.imshow(D)
plt.title('Distance Matrix')
io.show()
D_index = np.argsort(D, axis=1)
print D_index
indexOfRecognition = column(D_index, 0)
if trainingImage:
    indexOfRecognition = column(D_index, 1)
labelOfRecognition = []
for x in indexOfRecognition:
    labelOfRecognition.append(letters[x])
classesLetter = []
for x in classes:
    for y in x:
        classesLetter.append(y)
index = 0
if trainingImage:
    totalLabels = labelOfRecognition.__len__().__float__()
else:
    totalLabels = classes.__len__().__float__()
correctLabels = 0
print labelOfRecognition
if trainingImage:
    for x in labelOfRecognition:
        if x == fileLetter:
            correctLabels = correctLabels + 1
        index = index + 1
else:
    for x in labelOfRecognition:
        if index < classes.__len__():
            if x == truetestlabels[index]:
                correctLabels = correctLabels + 1
        index = index + 1
recognitionRate = correctLabels / totalLabels
print 'recognitionRate:', recognitionRate
