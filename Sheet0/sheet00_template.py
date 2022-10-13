import cv2 as cv
import numpy as np
import random
import sys
from numpy.random import randint


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows() 


if __name__ == '__main__':

    # set image path
    img_path = 'bonn.png' 
    
    # 2a: read and display the image 

    display_image('2 - a - Original Image', img)

    # 2b: display the intensity image

    display_image('2 - b - Intensity Image', img_gray)

    # 2c: for loop to perform the operation
    display_image('2 - c - Reduced Intensity Image', img_cpy)

    # 2d: one-line statement to perfom the operation above
    display_image('2 - d - Reduced Intensity Image One-Liner', img_cpy)    

    # 2e: Extract the center patch and place randomly in the image
    display_image('2 - e - Center Patch', img_patch)  
    
    # Random location of the patch for placement
    display_image('2 - e - Center Patch Placed Random %d, %d' % (rand_coord[0], rand_coord[1]), img_cpy)  

    # 2f: Draw random rectangles and ellipses
    display_image('2 - f - Rectangles and Ellipses', img_cpy)
       
    # destroy all windows
    cv.destroyAllWindows()
