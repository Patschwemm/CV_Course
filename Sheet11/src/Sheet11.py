import cv2
import numpy as np

def drawEpipolar(im1,im2,corr1,corr2,fundMat):

    ## Insert epipolar lines
    print("Drawing epipolar lines")
    cv2.imshow('Image 1', im1), \
    cv2.imshow('Image 2', im2), cv2.waitKey(0), cv2.destroyAllWindows()
    return

def display_correspondences(im1,im2,corr1,corr2):

    ## Insert correspondences
    print("Display correspondences")
    cv2.imshow('Image 1', im1), \
    cv2.imshow('Image 2', im2), cv2.waitKey(0), cv2.destroyAllWindows()
    return

def computeFundMat(im1,im2,corr1,corr2):
    fundMat = np.zeros((3,3))

    return fundMat

def question_q1_q2(im1,im2,correspondences):
    ## Compute and print Fundamental Matrix using the normalized corresponding points method.
    ## Display corresponding points and Epipolar lines
    corr1 = correspondences[:, :2]
    corr2 = correspondences[:, 2:]

    print("Compute Fundamental Matrix")
    fundMat = computeFundMat(im1.copy(),im2.copy(),corr1,corr2)
    display_correspondences(im1.copy(),im2.copy(),corr1,corr2)
    drawEpipolar(im1.copy(),im2.copy(),corr1,corr2,fundMat)
    return


def question_q3(im1, im2):
    dispar = np.zeros_like(im1)
    ## compute disparity map
    print("Compute Disparity Map")

    ## Display disparity Map
    cv2.imshow('Image 1', im1), \
    cv2.imshow('Image 2', im2), \
    cv2.imshow('Disparity Map', dispar), cv2.waitKey(0), cv2.destroyAllWindows()
    return

def question_q4(im1, im2, correspondences):
    corr1 = correspondences[:, :2]
    corr2 = correspondences[:, 2:]
    ## Perform Image rectification

    ### usage of either one is permitted
    print ("Fundamental Matrix")
    fundMat = np.asmatrix([[]]) ## Insert the given matrix
    fundMat = computeFundMat(im1.copy(),im2.copy(),corr1,corr2)

    ## Compute Rectification or Homography
    print("Compute Rectification")
    ## Apply Homography

    print("Display Warped Images")
    cv2.imshow('Warped Image 1', im1), \
    cv2.imshow('Warped Image 2', im2), cv2.waitKey(0), cv2.destroyAllWindows()
    return

def main():

    apt1 = cv2.imread('../images/apt1.jpg')
    apt2 = cv2.imread('../images/apt1.jpg')
    aloe1 = cv2.imread('../images/aloe1.png')
    aloe2 = cv2.imread('../images/aloe2.png')
    correspondences = np.genfromtxt('../images/corresp.txt', dtype=float, skip_header=1)
    question_q1_q2(apt1,apt2,correspondences)
    question_q3(aloe1,aloe2)
    question_q4(apt1,apt2,correspondences)

if __name__ == '__main__':
    main()
