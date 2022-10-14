from re import I
import cv2 as cv
import numpy as np
import random
import sys
import pathlib as Path
from numpy.random import randint


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    #cv.imshow(window_name, img)
    cv.imwrite("imgs/" + window_name + ".png", img)
    # cv.waitKey(0)
    # cv.destroyAllWindows() 


if __name__ == '__main__':

    # set image path
    img_path = 'bonn.png' 
    
    # 2a: read and display the image 

    # read image from path
    img = cv.imread(img_path)
    display_image('2 - a - Original Image', img)

    # 2b: display the intensity image

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    display_image('2 - b - Intensity Image', img_gray)

    # 2c: for loop to perform the operation

    # transpose so we can loop over channel first then x1 and x2
    img = img.transpose(2, 0, 1)
    print(img_gray.shape)
    # create copy of img to copy new values to new img
    img_cpy = np.empty_like(img)
    
    # loop over image
    for channel in range(img.shape[0]):
        for x1 in range(img.shape[1]):
            for x2 in range(img.shape[2]):
                img_cpy[channel, x1, x2] = img[channel, x1, x2] - img_gray[None, x1, x2] * 0.5

    # replace negative values
    img_cpy[img_cpy < 0] = 0

    # transpose back 
    img = img.transpose(1, 2, 0)
    img_cpy = img_cpy.transpose(1, 2, 0)


    display_image('2 - c - Reduced Intensity Image', img_cpy)

    # 2d: one-line statement to perfom the operation above

    # transpose new array again
    img_cpy = np.empty_like(img)
    img = img.transpose(2, 0, 1)

    # subtract intensity values
    img_cpy = img[:, None, :, :] - img_gray[None, None, :, :] * 0.5
    img_cpy[img_cpy < 0] = 0
    print(img_cpy[img_cpy < 0])

    # transpose back and squeeze new obsolete dimension
    img = img.transpose(1, 2, 0)
    print(img_cpy.shape)
    img_cpy = img_cpy.squeeze().transpose(1, 2, 0)

    display_image('2 - d - Reduced Intensity Image One-Liner', img_cpy)    

    # 2e: Extract the center patch and place randomly in the image

    img_patch = np.empty_like(img)
    display_image('2 - e - Center Patch', img_patch)  
    
    # Random location of the patch for placement
    display_image('2 - e - Center Patch Placed Random %d, %d' % (rand_coord[0], rand_coord[1]), img_cpy)  

    # 2f: Draw random rectangles and ellipses
    display_image('2 - f - Rectangles and Ellipses', img_cpy)
       
    # destroy all windows
    cv.destroyAllWindows()
