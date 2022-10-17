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
    # used imwrite because lack of graphical user interface
    #cv.imwrite("imgs/" + window_name + ".png", img)

    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows() 


if __name__ == '__main__':

    # set image path
    img_path = 'bonn.png' 
    
    ###############################################################
    ###############################################################

    # 2a: read and display the image 

    # read image from path
    img = cv.imread(img_path)
    display_image('2 - a - Original Image', img)

    ###############################################################
    ###############################################################

    # 2b: display the intensity image

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    display_image('2 - b - Intensity Image', img_gray)

    ###############################################################
    ###############################################################

    # 2c: for loop to perform the operation

    #conversion from uint to normal int necessary for negative values not to become 255 again
    img_int = img.copy().astype(np.int)

    # transpose so we can loop over channel first then x1 and x2
    img_int = img_int.transpose(2, 0, 1)
    print(img_gray.shape)
    # create copy of img_int to copy new values to new img_int
    img_cpy = np.empty_like(img_int)
    print(type(img_int), print(type(img_cpy)))
    
    # loop over image
    for channel in range(img_int.shape[0]):
        for x1 in range(img_int.shape[1]):
            for x2 in range(img_int.shape[2]):
                img_cpy[channel, x1, x2] = img_int[channel, x1, x2] - img_gray[None, x1, x2] * 0.5

    # replace negative values
    img_cpy[img_cpy < 0] = 0

    # transpose back 
    img_int = img_int.transpose(1, 2, 0)
    img_cpy = img_cpy.transpose(1, 2, 0)
    print(type(img_int[0,0,0]))

    display_image('2 - c - Reduced Intensity Image', img_cpy) # strange behaviour with the pixels

    ###############################################################
    ###############################################################

    # 2d: one-line statement to perfom the operation above

    # transpose new array again
    img_cpy = np.empty_like(img)
    img = img.transpose(2, 0, 1)

    # subtract intensity values
    img_cpy = img[:, None, :, :] - img_gray[None, None, :, :] * 0.5
    img_cpy[img_cpy < 0] = 0

    # transpose back and squeeze new obsolete dimension
    img = img.transpose(1, 2, 0)
    img_cpy = img_cpy.squeeze().transpose(1, 2, 0)

    display_image('2 - d - Reduced Intensity Image One-Liner', img_cpy)   

    ###############################################################
    ################################################################ 

    # 2e: Extract the center patch and place randomly in the image

    # get img center
    center = (img.shape[0]//2, img.shape[1]//2)
    #define patch size
    patch_size = (16, 16)
    print(center)
    # create placeholder array for patch
    img_patch = np.zeros((16, 16, 3))
    # extract the patch
    img_patch = img[center[0] - patch_size[0]//2 : center[0] + patch_size[0]//2,  center[1] - patch_size[1]//2 : center[1] + patch_size[1]//2, :]
    display_image('2 - e - Center Patch', img_patch)  
    
    # Random location of the patch for placement

    #generate random coordinates
    rand_coord = (randint(0, img.shape[0]) , randint(0, img.shape[1]))

    # copy img
    img_patch_inserted = np.copy(img)
    # insert patch into copied img 
    img_patch_inserted[rand_coord[0] - patch_size[0]//2 : rand_coord[0] + patch_size[0]//2, rand_coord[1] - patch_size[1]//2 : rand_coord[1] + patch_size[1]//2, :] = img_patch

    display_image('2 - e - Center Patch Placed Random %d, %d' % (rand_coord[0], rand_coord[1]), img_patch_inserted)  

    # 2f: Draw random rectangles and ellipses

    # empty image
    image_primitives = np.zeros((1000, 1000, 3))

    # define basic colors
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

    # only do rectangles because ellipses is just looking in documentary and more random variables to create random ellipses...
    for i in range(10):
        #create 2 corners to span rectangle, random color and set thickness to -1 to fill img
        corner1 = (randint(0, image_primitives.shape[0]) , randint(0, image_primitives.shape[1]))
        corner2 = (randint(0, image_primitives.shape[0]) , randint(0, image_primitives.shape[1]))
        rand_color = colors[randint(0, 3)]
        cv.rectangle(image_primitives, corner1, corner2, rand_color, thickness = -1)


    display_image('2 - f - Rectangles and Ellipses', image_primitives)
       
    # destroy all windows
    cv.destroyAllWindows()
