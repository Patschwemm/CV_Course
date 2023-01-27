import os
import cv2
import numpy as np
import random

def project(points, intrinsic, pos):

    Rt = np.eye(3, 4)
    Rt[:, 3] = -pos


    K = np.eye(3)
    K[0, 0] = intrinsic[0]
    K[1, 1] = intrinsic[1]
    K[0, 2] = intrinsic[2]
    K[1, 2] = intrinsic[3]

    P = np.matmul(K, Rt)
    ones = np.ones(points.shape[0]).reshape(-1, 1)
    points3d = np.append(points, ones, 1).transpose()
    points2d = np.matmul(P, points3d).transpose()
    points2d = points2d[:, :2]/points2d[:,2:]

    return points2d

def drawEpipolar(corr1,corr2,im,fundMat):

    for i in range(corr1.shape[0]):
        leftP = np.array([corr1[i,0],corr1[i,1],1]).reshape(3,1)
        rightP = np.matmul(fundMat,leftP)
        color = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))

        for j in range(0,im.shape[0]):
            ptR = (j+1,int((-rightP[2,0]-rightP[0,0]*(j+1))/rightP[1,0]))
            cv2.circle(im,ptR,1,color)

        cv2.circle(im, (int(corr2[i, 0]), int(corr2[i, 1])), 3, color, thickness=2, lineType=8, shift=0)

    cv2.imwrite("epipolar.png", im)
    return


def computeFundMat(corr1, corr2):
    fundMat = np.zeros((3,3))
    mean1 = np.mean(corr1, 0, dtype=float) # (2, )
    mean2 = np.mean(corr2, 0, dtype=float) # (2, )

    rmsd1 = np.mean(np.sqrt(np.power(corr1[:, 0] - mean1[0], 2) + np.power(corr1[:, 1] - mean1[1], 2)))
    rmsd2 = np.mean(np.sqrt(np.power(corr2[:, 0] - mean2[0], 2) + np.power(corr2[:, 1] - mean2[1], 2)))


    ncorr1 = (corr1 - mean1) * np.sqrt(2) / rmsd1
    ncorr2 = (corr2 - mean2) * np.sqrt(2) / rmsd2


    A = np.zeros((ncorr1.shape[0],9),dtype=float)
    A[:, 0] = ncorr1[:, 0] * ncorr2[:, 0]
    A[:, 1] = ncorr1[:, 1] * ncorr2[:, 0]
    A[:, 2] = ncorr2[:, 0]
    A[:, 3] = ncorr1[:, 0] * ncorr2[:, 1]
    A[:, 4] = ncorr1[:, 1] * ncorr2[:, 1]
    A[:, 5] = ncorr2[:, 1]
    A[:, 6] = ncorr1[:, 0]
    A[:, 7] = ncorr1[:, 1]
    A[:, 8] = 1

    U, s, V = np.linalg.svd(A)
    f = V[8,:].reshape(3,3)
    U, s, V = np.linalg.svd(f)

    ## enforce rank 2
    s_new = np.zeros((3,3),dtype=float)
    s_new[0, 0] = s[0]
    s_new[1, 1] = s[1]

    f_rank2 = np.dot(U,np.dot( s_new , V))

    Tl = np.array([np.sqrt(2) / rmsd1, 0, -1 * mean1[0] * np.sqrt(2) / rmsd1, 0, np.sqrt(2) / rmsd1,
                   -1 * mean1[1] * np.sqrt(2) / rmsd1, 0, 0, 1]).reshape((3,3))
    Tr = np.array([np.sqrt(2) / rmsd2, 0, -1 * mean2[0] * np.sqrt(2) / rmsd2, 0, np.sqrt(2) / rmsd2,
          -1 * mean2[1] * np.sqrt(2) / rmsd2, 0, 0, 1]).reshape((3,3))
    f_rank2 = np.dot(Tr.transpose(), np.dot(f_rank2, Tl))
    fundMat = f_rank2
    return fundMat

def main():

    points = np.genfromtxt('data/3d_points.txt', dtype=float, skip_header=1)

    fx = 525.0  # focal length x
    fy = 525.0  # focal length y
    cx = 319.5  # optical center x
    cy = 239.5  # optical center y

    intrinsic = (fx, fy, cx, cy)

    pos = np.array([2, -3, 1])

    points2d = project(points, intrinsic, pos)

    print("Projected points are : ")
    print(points2d)

    correspondences = np.genfromtxt('data/2d_corresp.txt', dtype=float, skip_header=1)

    corr1 = correspondences[:, :2]
    corr2 = correspondences[:, 2:]

    fundMat = computeFundMat(corr1, corr2)
    print("Fundamental Matrix is : \n", fundMat)

    apt = cv2.imread('./data/apt.jpg')

    drawEpipolar(corr1,corr2,apt.copy(),fundMat)

if __name__ == '__main__':
    main()