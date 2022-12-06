#!/usr/bin/python3.5

import numpy as np
import cv2 as cv

'''
    load the image and foreground/background parts
    image: the original image
    background/foreground: numpy array of size (n_pixels, 3) (3 for RGB values), i.e. the data you need to train the GMM
'''


def read_image(filename):
    image = cv.imread(filename) / 255
    height, width = image.shape[:2]
    bounding_box = np.zeros(image.shape)
    bounding_box[75:490, 130:250, :] = 1
    foreground = image[bounding_box == 1].reshape((-1, 3))
    background = image[bounding_box == 0].reshape((-1, 3))
    return image, foreground, background

if __name__ == '__main__':

    image, foreground, background = read_image('data/cv_is_great.png')

    '''
    TODO: compute p(x|w=background) for each image pixel and manipulate the image such that everything below the threshold is black, display the resulting image
    Hint: Slide 64
    '''
