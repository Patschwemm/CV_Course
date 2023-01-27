import cv2
import numpy as np
import random


def computeFundMat(im1,im2,corr1,corr2):
    fundMat = np.zeros((3,3))

    return fundMat


def question_q2(im1, im2):
    dispar = np.zeros_like(im1[:,:,0],dtype=np.int8)
    wsize = 15
    ## compute disparity map
    print("Compute Disparity Map")

    for i in range(im1.shape[0]-wsize):
        haystack = im2[i:(i+wsize),0:im1.shape[1],:]
        for j in range(im1.shape[1]-wsize):
            needle = im1[i:(i+wsize),j:(j+wsize),:]
            matchARR = cv2.matchTemplate(haystack,needle,cv2.TM_SQDIFF)
            a = cv2.minMaxLoc(matchARR)

            matchLoc = (a[2][0],j+int(wsize/2.0))

            dispar[i, j] = int(np.sqrt(
                np.power(i + int(wsize / 2.0) - matchLoc[0], 2) + np.power(j + int(wsize / 2.0) - matchLoc[1], 2)))

        # print(im1.shape[0], i, wsize, haystack.shape)
        # cv2.imshow("",haystack), cv2.waitKey(0)


    # cv2.normalize(dispar,dispar,0,1,cv2.NORM_MINMAX)
    # print (dispar)
    ## Display disparity Map
    cv2.imshow('Image 1', im1), \
    cv2.imshow('Image 2', im2), \
    cv2.imshow('Disparity Map', dispar), cv2.waitKey(0), cv2.destroyAllWindows()
    return

def question_q3(im1, im2, correspondences):
    corr1 = correspondences[:, :2]
    corr2 = correspondences[:, 2:]
    im1 = im1[:, :, 0].copy()
    im2 = im2[:, :, 0].copy()
    ## Perform Image rectification

    ### usage of either one is permitted
    print ("Fundamental Matrix")
    fundMat = np.asmatrix([[-1.78999e-7,5.70878e-6,-0.00260653,-5.71422e-6,1.63569e-7,-0.0068799,0.00253316,0.00674493,.191989]]).reshape(3,3) ## Insert the given matrix
    #fundMat = computeFundMat(im1.copy(),im2.copy(),corr1,corr2)

    # compute left epipolar
    # compute the left epipole

    U, s, V = np.linalg.svd(fundMat)

    left_ep = (V[2,:]/V[2,0]).reshape(1,3)

    ## Compute Rectification or Homography
    print("Compute Rectification")

    w = im1.shape[1]
    h = im1.shape[0]

    PPt = np.array([w*w-1, 0, 0, 0, h*h -1, 0, 0, 0, 0],dtype=np.float).reshape((3,3))
    pcpct = np.array([(w-1)*(w-1), (w-1)*(h-1), 2*(w-1), (w-1)*(h-1), (h-1)*(h-1), 2*(h-1), 2*(w-1), 2*(h-1), 4]
                     ,dtype=np.float).reshape((3,3))
    PPt *= (w*h/12)
    pcpct /= 4

    ex = np.array([0, -left_ep[0,2], left_ep[0,1], left_ep[0,2], 0, -left_ep[0,0], -left_ep[0,1], left_ep[0,0], 0]).reshape((3,3))

    temp = np.dot(ex.transpose(),np.dot(PPt,ex))
    A = temp[:2,:2]

    temp = np.dot(ex.transpose(),np.dot(pcpct,ex))
    B = temp[:2, :2]

    temp = np.dot(fundMat.transpose(),np.dot(PPt, fundMat))
    Ap = temp[:2, :2]

    temp = np.dot(fundMat.transpose(),np.dot(pcpct, fundMat))
    Bp = temp[:2, :2]

    U,sA,V = np.linalg.svd(A)
    sA = np.sqrt(sA)
    temp = np.matrix([[sA[0],0],[0,sA[1]]])
    D = np.dot(U, temp).transpose()

    temp = np.linalg.inv(D).transpose() * B * np.linalg.inv(D)
    e,v = np.linalg.eig(temp)
    z1 = np.dot(np.linalg.inv(D),v[:,0])
    z1 /= np.sqrt(np.power(z1,2).sum())

    U, sA, V = np.linalg.svd(Ap)
    sA = np.sqrt(sA)
    temp = np.matrix([[sA[0], 0], [0, sA[1]]])
    Dp = np.dot(U, temp).transpose()

    temp = np.linalg.inv(Dp).transpose() * Bp * np.linalg.inv(Dp)
    e, v = np.linalg.eig(temp)
    z2 = np.dot(np.linalg.inv(Dp), v[:, 0])
    z2 /= np.sqrt(np.power(z2, 2).sum())

    z = np.zeros((3,1),dtype=np.float)
    z[:2,:] = (z1 + z2)/2

    w_vec = np.dot(ex, z)
    wp_vec = np.dot(fundMat, z)

    Hp = np.zeros((3,3),dtype=np.float)
    Hp[0, 0] = 1
    Hp[1, 1] = 1
    Hp[2, :] = w_vec.squeeze()
    Hpp = np.zeros((3, 3), dtype=np.float)
    Hpp[0, 0] = 1
    Hpp[1, 1] = 1
    Hpp[2, :] = wp_vec.squeeze()


    Hr = np.zeros((3,3),dtype=np.float)
    Hr[0, 0] = fundMat[2, 1] - Hp[2, 1] * fundMat[2, 2]
    Hr[0, 1] = Hp[2, 0] * fundMat[2, 2] - fundMat[2, 0]
    Hr[1, 0] = fundMat[2, 0] - Hp[2, 0] * fundMat[2, 2]
    Hr[1, 1] = fundMat[2, 1] - Hp[2, 1] * fundMat[2, 2]
    Hr[1, 2] = fundMat[2, 2]
    Hr[2, 2] = 1

    Hrp = np.zeros((3, 3), dtype=np.float)
    Hrp[0, 0] = Hpp[2, 1] * fundMat[2, 2] - fundMat[1, 2]
    Hrp[0, 1] = fundMat[0, 2] - Hpp[2, 1] * fundMat[2, 2]
    Hrp[1, 0] = Hpp[2, 0] * fundMat[2, 2] - fundMat[0, 2]
    Hrp[1, 1] = Hpp[2, 1] * fundMat[2, 2] - fundMat[1, 2]
    Hrp[1, 2] = 0
    Hrp[2, 2] = 1
    ## Apply Homography
    scale = .05
    minvcp = 1
    maxvcp = 1
    for i in range(im1.shape[0]):
        for j in range(im1.shape[1]):
            temp_vec = [(i+1),(j+1),1]
            map_loc = scale * np.dot(Hr, np.dot(Hp , temp_vec))
            map_loc /= map_loc[2]

            if map_loc[1] < minvcp:
                minvcp = int(map_loc[1])


            map_loc = scale * np.dot(Hrp, np.dot(Hpp, temp_vec))
            map_loc /= map_loc[2]

            if map_loc[1] < minvcp:
                minvcp = int(map_loc[1])

    
    Hr[1,2] -= minvcp
    Hrp[1, 2] -= minvcp
    
    # deprecated
    i1 = np.zeros((3010,1900),dtype=np.int8)
    i2 = np.zeros((200,200),dtype=np.int8)
    x = 270817
    for i in range(im1.shape[0]):
        for j in range(im1.shape[1]):
            temp_vec = [(i + 1), (j + 1), 1]
            map_loc = scale * np.dot(Hp, temp_vec)
            map_loc /= map_loc[2]
            tempx = int(map_loc[0])
            tempy = int(map_loc[1])
            if tempx >= 1 and tempx <= i1.shape[0] and tempy >= 1 and tempy <= i1.shape[1]:
                i1[tempx -1,tempy-1] = im1[i,j]

            map_loc = scale * np.dot(Hpp, temp_vec)
            map_loc /= map_loc[2]

            tempx = int((map_loc[0] - 756)/1000) ## Offset and normalization
            tempy = int((map_loc[1] -568)/1000) ## Offset and normalization
            x = min(x, tempx)
            if tempx >= 1 and tempx <= i2.shape[0] and tempy >= 1 and tempy <= i2.shape[1]:
                i2[tempx - 1, tempy - 1] = im2[i, j]

    Hr[1, 2] += minvcp
    Hrp[1, 2] += minvcp
    print (x)

    print("Display Warped Images")
    cv2.imshow('Warped Image 1', i1), \
    cv2.imshow('Warped Image 2', i2), cv2.waitKey(0), cv2.destroyAllWindows()
    return

def main():

    apt1 = cv2.imread('./images/apt1.jpg')
    apt2 = cv2.imread('./images/apt2.jpg')
    aloe1 = cv2.imread('./images/aloe1.png')
    aloe2 = cv2.imread('./images/aloe2.png')
    correspondences = np.genfromtxt('./images/corresp.txt', dtype=float, skip_header=1)
    question_q2(aloe1,aloe2)
    question_q3(apt1,apt2,correspondences)

if __name__ == '__main__':
    main()
