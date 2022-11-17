import numpy as np
import cv2 as cv
# import random
# from collections import defaultdict

##############################################
#     Task 1        ##########################
##############################################


def task_1_a():
    print("Task 1 (a) ...")
    img = cv.imread('../images/billiards.png')
    '''
    ...
    your code ...
    ...
    '''


def myHoughCircles(edges, minRadius, maxRadius, threshold, minDist):
    """
    Your implementation of HoughCircles
    :edges: single-channel binary source image (e.g: edges)
    :minRadius: minimum circle radius
    :maxRadius: maximum circle radius
    :param threshold: minimum number of votes to consider a detection
    :minDist: minimum distance between two centers of the detected circles. 
    :return: list of detected circles as (a, b, r) triplet
    """
    # accumulator = ...
    detected_circles = []
    '''
    ...
    your code ...
    ...
    '''
    return detected_circles


def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread('../images/billiards.png')
    minRadius = 10
    maxRadius = 100
    minDist = img.shape[0]/8
    threshold = 40
    # resolution = 1
    img_gray = None # convert the image into grayscale
    edges = None # detect the edges
    detected_circles = myHoughCircles(edges, minRadius, maxRadius, threshold, minDist)
    '''
    ...
    your code ...
    ...
    '''


##############################################
#     Task 2        ##########################
##############################################

def houghLines(img_edges, d_resolution, theta_step_sz, threshold):
    """
    Implementation of HoughLines
    :param img_edges: single-channel binary source image (e.g: edges)
    :param d_resolution: the resolution for the distance parameter
    :param theta_step_sz: the resolution for the angle parameter
    :param threshold: minimum number of votes to consider a detection
    :return: list of detected lines as (d, theta) pairs and the accumulator
    """
    accumulator = np.zeros((int(180/theta_step_sz), int(np.linalg.norm(img_edges.shape)/d_resolution)))
    edges_points = np.array(np.nonzero(img_edges))

    for i in range(edges_points.shape[1]):
        for theta in range(0, 180, theta_step_sz):
            d = int((edges_points[1][i] * np.cos(theta*np.pi/180.) + edges_points[0][i] * np.sin(theta*np.pi/180.)) / d_resolution)
            accumulator[int(theta/theta_step_sz), d] += 1
    
    accumulator_copy = accumulator
    detected_lines = []
    finished = False
    while not finished:
        idx = np.argmax(accumulator_copy)
        theta, d = np.unravel_index(idx, accumulator_copy.shape)

        if accumulator_copy[theta, d] > threshold:
            detected_lines.append([d * d_resolution, theta * theta_step_sz * np.pi / 180.])
        else:
            finished = True

        accumulator_copy[theta, d] = 0

    return detected_lines, accumulator


def task_2():
    print("Task 2 ...")
    img = cv.imread('../images/line.png')
    img_gray = None # convert the image into grayscale
    edges = None # detect the edges
    theta_res = None # set the resolution of theta
    d_res = None # set the distance resolution
    #_, accumulator = houghLines(edges, d_res, theta_res, 50)
    '''
    ...
    your code ...
    ...
    '''


##############################################
#     Task 3        ##########################
##############################################


def myKmeans(data, k):
    """
    Your implementation of k-means algorithm
    :param data: list of data points to cluster
    :param k: number of clusters
    :return: centers and list of indices that store the cluster index for each data point
    """
    centers = np.zeros((k, data.shape[1]))
    index = np.zeros(data.shape[0], dtype=int)
    clusters = [[] for i in range(k)]

    # initialize centers using some random points from data
    # ....

    convergence = False
    iterationNo = 0
    while not convergence:
        # assign each point to the cluster of closest center
        # ...

        # update clusters' centers and check for convergence
        # ...

        iterationNo += 1
        print('iterationNo = ', iterationNo)

    return index, centers


def task_3_a():
    print("Task 3 (a) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


def task_3_b():
    print("Task 3 (b) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


def task_3_c():
    print("Task 3 (c) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


##############################################
#     Task 4        ##########################
##############################################


def task_4_a():
    print("Task 4 (a) ...")
    D = None  # construct the D matrix
    W = None  # construct the W matrix
    '''
    ...
    your code ...
    ...
    '''


##############################################
##############################################
##############################################

if __name__ == "__main__":
    task_1_a()
    task_1_b()
    task_2()
    task_3_a()
    task_3_b()
    task_3_c()
    task_4_a()

