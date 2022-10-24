import cv2
import numpy as np
import os
import time

DATA_DIR = '../data'
SIM_THRESHOLD = 0.5 # similarity threshold for template matching. Can be adapted.

# blur the image in the spatial domain using convolution
def blur_im_spatial(image, kernel_size):
    #TODO
    pass
    

# blur the image in the frequency domain
def blur_im_freq(image, kernel):
    #TODO
    pass
 
# implement the sum square difference (SQD) similarity 
def calc_sum_square_difference(image, template):
    pass
       
# implement the normalized cross correlation (NCC) similarity 
def calc_normalized_cross_correlation(image, template):
    pass

#draw rectanges on the input image in regions where the similarity is larger than SIM_THRESHOLD
def draw_rectangles(input_im, similarity_im):
    pass

#You can choose to resize the image using the new dimensions or the scaling factor
def pyramid_down(image, dstSize, scale_factor=None):   
    pass
#create a pyramid of the image using the specified pyram function pyram_method.
#pyram_func can either be cv2.pyrDown or your own implementation
def create_gaussian_pyramid(image, pyram_func, num_levels):
    #in a loop, create a pyramid of downsampled blurred images using the Gaussian kernel
    pass
def calc_derivative_gaussian_kernel(size, sigma):
    # TODO: implement
    pass

def create_laplacian_pyramid(image, num_levels=5):
    #create the laplacian pyramid using the gaussian pyramid
    gaussian_pyramid = create_gaussian_pyramid(image, cv2.pyrdown, num_levels)
    #complete as described in the exercise sheet
    pass
# Given the final weighted pyramid, sum up the images at each level with the upscaled previous level
def collapse_pyramid(laplacian_pyramid):
    
    final_im = laplacian_pyramid[0]
    for l in range(1, len(laplacian_pyramid)):
        #TODO complete code 
        pass
    return final_im
#Fourier Transform

def task1(input_im_file):
    full_path = os.path.join(DATA_DIR, input_im_file)
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    kernel_siZe = 7
    kernel = None  # TODO: create kernel
    # time the blurring of the different methods
    start_time = time.time()
    conv_result = blur_im_spatial(image, kernel_siZe) 
    end_time = time.time()
    print('time taken to apply blur in the spatial domain', end_time-start_time)
    # measure the timing here too
    fft_result = blur_im_freq(image, kernel)

    # TODO: compare results in terms of run time and mean square difference




#Template matching using single-scale
def task2(input_im_file, template_im_file):
    full_path_im = os.path.join(DATA_DIR, input_im_file)
    full_path_template = os.path.join(DATA_DIR, template_im_file)
    in_im = cv2.imread(full_path_im, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_im_file, cv2.IMREAD_GRAYSCALE)
    result_sqd = calc_sum_square_difference(in_im, template)
    result_ncc = calc_normalized_cross_correlation(in_im, template)

    #draw rectanges at matching regions
    vis_sqd = draw_rectangles(in_im, result_sqd)
    vis_ncc = draw_rectangles(in_im, result_ncc)
    


def task3(input_im_file, template_im_file):
    pass
    # TODO: calculate the time needed for template matching with the pyramid

    # TODO: show the template matching results using the pyramid



#Image blending
def task4(input_im_file1, input_im_file2, interest_region_file, num_pyr_levels=5):
    #TODO you can use the steps described in the exercise sheet to help guide you through the solution
    result = None
    return result

def task5(input_im, kernel_size=5, sigma=0.5):
    image = cv2.imread("../data/einstein.jpeg", 0)

    kernel_x, kernel_y = calc_derivative_gaussian_kernel(kernel_size, sigma)

    edges_x = None  # TODO: convolve with kernel_x
    edges_y = None  # TODO: convolve with kernel_y

    magnitude = None  # TODO: compute edge magnitude
    direction = None  # TODO: compute edge direction

    # TODO visualise the results



if __name__ == "__main__":
    task1('orange.jpeg')
    task1('celeb.jpeg')
    task2('RidingBike.jpeg', 'RidingBikeTemplate.jpeg')
    task3('DogGray.jpeg', 'DogTemplate.jpeg')
    task4('dog.jpeg', 'moon.jpeg', 'mask.jpeg')
    # just for fun, blend these these images as well
    for i in range[1,2,10]:
        ind = str(i).zfill(2)
        blended_im = task4('task4_extra/source_%s.jpg'%ind, 'task4/target_%s.jpg'%ind, 'task4/mask_%s.jpg'%ind)
        #visualise the blended image

    task5('einstein.jpeg')