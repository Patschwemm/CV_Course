import cv2
import numpy as np
import os
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy.linalg

DATA_DIR = '../data/task%s'

#
def decomposePCA(dataPoints,  k=None, preservRatio=None):
    # implement PCA for task1 yourself and return the first k 
    # components that preserve preservRatio of the energy and their eigen valuess
    pass

def loadData(filename):
    #return the data points
    df = pd.read_csv(filename, header = 0)
    pass

def visualiseHands(kpts, title):
    #use matplotlib for that
    pass


#task 1: training the statistical shape model
# return the trained model so you can use it in task 2
def task1(train_file='hands_aligned_train.txt'):

    trainfilePath = os.path.join(DATA_DIR%'1', train_file)
    




#task2: performing inference on the test hand data using the trained shape model
def task2(shapeModel, test_file='hands_aligned_test.txt'):
    testfilePath = os.path.join(DATA_DIR%'1', test_file)

    
    

#eigen faces
def task3():
    #(a)you can use scikit PCA for the decomposition and the dataset class for loading LFW dataset directly
    # train over LFW training split that you created
    #(b) evaluate over the samples in data/task3/detect/face and data/task3/detect/notFace
    
    positiveFilenames = os.listdir(DATA_DIR%'3', 'detect/face')
    negativeFilenames = os.path.listdir(DATA_DIR%'3', 'detect/notFace')
    #calculate the accuracy
    
    #(c) test over LFW test split and calculate the accuracy of recognition
    pass

#compute the structural tensor M, you can apply an opencv filters to calculate the gradients
def computeStructural(image):
    M = None
    #todo
    return M
def detectorCornerHarris(image, M, responseThresh):
    pass

def detectorCornerFoerstner(image, M, responseThresh):
    pass

# corner detectors: implement Harris corner detector and Förstner corner  
def task4(imFile ='palace.jpeg'):
    image = cv2.imread(DATA_DIR%'4', imFile)
    
    #(a)
    #todo
    #(b) apply Harris corner detector and visualise
    #todo
    #(c) apply Foerstner corner detector and visualise
    #todo
    pass

# perform the matching using sift
def match(sift, image1, image2):
    pass
def rotateImage(image, rotAngle):
    # to get the rotation matrix, you can use cv2.getRotationMatrix2D
    rotMat = None
    rotatedIm = cv2.warpAffine(image, rotMat) #fill in the missing parameters
    return rotatedIm

#keypoint matching
def task5(imFile = 'castle.jpeg'):
    
    
    image = cv2.imread(DATA_DIR%'5'%imFile)
    #you can use cv2's implementation of SIFT
    sift = cv2.SIFT_create()
    #apply each of the following transformations and then perform matching
    # Please choose two values of your choice for each one
    rotAngles = [0, 0] #todo choose two values
    for rotAngle in rotAngles:
        rotatedIm = rotateImage(image, rotAngle)
        
        match(sift, image, rotatedIm)
        #todo visualise using cv2.drawMatches
    
    #do the same with a translation transformation
    
    #do the same with a scaling transformation
 
#Image homography: implement the RANSAC algorithm then apply it to stitch the two images of Bonn's
# Poppelsdorfer Schloss and then visualise the stitched image
def task6():
    pass
    

if __name__ == "__main__":
    handModel = task1()
    task2(handModel)
    task3()
    task4()
    task5()
    task6()
    