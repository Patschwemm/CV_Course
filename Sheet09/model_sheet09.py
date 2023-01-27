import numpy as np
import os
import cv2 as cv


EIGEN_THRESHOLD = 0.01
EPSILON= 0.002
MAX_ITERS = 1000
UNKNOWN_FLOW_THRESH = 1000

def load_FLO_file(filename):
    assert os.path.isfile(filename), 'file does not exist: ' + filename   
    flo_file = open(filename,'rb')
    magic = np.fromfile(flo_file, np.float32, count=1)
    assert magic == 202021.25,  'Magic number incorrect. .flo file is invalid'
    w = np.fromfile(flo_file, np.int32, count=1)
    h = np.fromfile(flo_file, np.int32, count=1)
	#if error try: data = np.fromfile(flo_file, np.float32, count=2*w[0]*h[0])
    #the float values for u and v are interleaved in row order, i.e., u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...,
    # in total, there are 2*w*h flow values
    data = np.fromfile(flo_file, np.float32, count=2*w[0]*h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h[0]), int(w[0]), 2))
    flo_file.close()
    return flow



#function for converting flow map to to BGR image for visualisation
def flow_map_to_bgr(flow):
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    height, width, chan = flow.shape
    hsv = np.zeros((height, width, 3))
    
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,1] = 255
    hsv[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX) 
    
    bgr = cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR)
    return bgr

#***********************************************************************************
#implement Lucas-Kanade Optical Flow 
# Parameters:
# frames: the two consecutive frames
# Ix: Image gradient in the x direction
# Iy: Image gradient in the y direction
# It: Image gradient with respect to time
# window_size: the number of points taken in the neighborhood of each pixel
# returns the Optical flow based on the Lucas-Kanade algorithm
def Lucas_Kanade_flow(frames, Ix, Iy, It, window_size, based_on_eigen=True):

    IxIx_neighbors_sum = Ix**2
    IyIy_neighbors_sum = Iy**2
    IxIy_neighbors_sum = Ix * Iy
    
    IxIt_neighbors_sum = Ix * It
    IyIt_neighbors_sum = Iy * It

    IxIx_neighbors_sum = cv.boxFilter(IxIx_neighbors_sum, -1, (window_size[0], window_size[1]), normalize=False) # unnormalized to get the sum rather the average
    IyIy_neighbors_sum = cv.boxFilter(IyIy_neighbors_sum, -1, (window_size[0], window_size[1]), normalize=False)
    IxIy_neighbors_sum = cv.boxFilter(IxIy_neighbors_sum, -1, (window_size[0], window_size[1]), normalize=False)
    IxIt_neighbors_sum = cv.boxFilter(IxIt_neighbors_sum, -1, (window_size[0], window_size[1]), normalize=False)
    IyIt_neighbors_sum = cv.boxFilter(IyIt_neighbors_sum, -1, (window_size[0], window_size[1]), normalize=False)
    
    w, h = I1.shape[1], I1.shape[0]
    flow = np.zeros((h, w, 2))
    for r in range(h):
        for c in range(w):
            A = np.array([ [IxIx_neighbors_sum[r,c], IxIy_neighbors_sum[r,c] ] ,[IxIy_neighbors_sum[r,c], IyIy_neighbors_sum[r,c]] ])
            b = np.array([ [-IxIt_neighbors_sum[r,c]], [-IyIt_neighbors_sum[r,c]] ])
            
            if based_on_eigen:
                eigen_vals = np.linalg.eigvals(A)
                min_eigen_val = np.min(eigen_vals)
                if min_eigen_val < EIGEN_THRESHOLD: # the flow is not valid for this pixel
                    print('invalid flow at this point!!!')
                    print(eigen_vals)
                    continue
                A_inv = np.linalg.inv(A)
            else:
                A_inv = cv.invert(A, cv.DECOMP_SVD)[1]
            flow[r,c,:] = np.dot(A_inv, b)[:, 0]
           
    return flow   


#***********************************************************************************
#implement Horn-Schunck Optical Flow 
# Parameters:
# frames: the two consecutive frames
# Ix: Image gradient in the x direction
# Iy: Image gradient in the y direction
# It: Image gradient with respect to time
# alpha: smoothness term, try different values to see its influence
# returns the Optical flow based on the Horn-Schunck algorithm
def Horn_Schunck_flow(Ix, Iy, It, alpha=1.):
    
    denom = alpha**2+Ix**2+Iy**2
    u = np.zeros( (Ix.shape[0], Ix.shape[1]) )
    v = np.zeros( (Ix.shape[0], Ix.shape[1]) )
    flow_horn_schunck = np.zeros( (Ix.shape[0], Ix.shape[1], 2) )
    diff = 1
    iter = 0
    while diff > EPSILON and iter < MAX_ITERS:
        u_laplace = cv.Laplacian(u, -1, ksize=1, scale=0.25)
        v_laplace = cv.Laplacian(v, -1, ksize=1, scale=0.25)
        u_mean = u + u_laplace
        v_mean = v + v_laplace
        mult_term = (Ix*u_mean + Iy*v_mean + It)/denom
        prev_u = u.copy()
        prev_v = v.copy()
        u = u_mean - Ix * mult_term
        v = v_mean - Iy * mult_term
        diff = cv.norm(prev_u, u, normType=cv.NORM_L2) + cv.norm(prev_v, v, normType=cv.NORM_L2)
        iter += 1
    print('number of iterations, difference: %d, %.4f' %(iter, diff))
    flow_horn_schunck[:,:,0] = u
    flow_horn_schunck[:,:,1] = v
    return flow_horn_schunck



def calculate_angular_error(estimated_flow, groundtruth_flow):
    aae = groundtruth_flow[:,:,0]*estimated_flow[:,:,0]+groundtruth_flow[:,:,1]*estimated_flow[:,:,1] + 1
    denom = np.sqrt( (groundtruth_flow[:,:,0]**2+groundtruth_flow[:,:,1]**2+1) * (estimated_flow[:,:,0]**2+estimated_flow[:,:,1]**2+1) )
    aae = aae/denom
    aae = ( np.sum(np.arccos(aae)) ) / aae.size
    aae = aae * 180 / np.pi #convert to degrees 
    return aae




if __name__ == "__main__":
    #load ground truth of the optical flow
    groundtruth_flow = load_FLO_file('./data/groundTruthOF.flo')
    groundtruth_flow[groundtruth_flow > 10 ** 3] = 0

    #load the images
    I1 = cv.cvtColor(cv.imread('data/frame1.png'), cv.COLOR_BGR2GRAY)
    I2 = cv.cvtColor(cv.imread('data/frame2.png'), cv.COLOR_BGR2GRAY)
    frames = np.float32(np.array([I1, I2]))
    frames /= 255.0

   
    #calculate image gradient
    Ix = cv.Sobel(frames[0], cv.CV_32F, 1, 0, 3)
    Iy = cv.Sobel(frames[0], cv.CV_32F, 0, 1, 3)
    It = frames[1]-frames[0]
    
    #applying Lucas Kanade Optical Flow 
    WINDOW_SIZE = [15, 15]  #the number of points taken in the neighborhood of each pixel when applying Lucas-Kanade
    if True:
        flow_lucas_kanade = Lucas_Kanade_flow(frames, Ix, Iy, It, WINDOW_SIZE)
        flow_lucas_kanade_bgr = flow_map_to_bgr(flow_lucas_kanade)

        aae_lucas_kanade = calculate_angular_error(flow_lucas_kanade, groundtruth_flow)

        print('Average Angular error for Luacas-Kanade Optical Flow: %.4f' %(aae_lucas_kanade))
        
    # Applying Horn Schunck Optical Flow
    if True:
        flow_horn_schunck = Horn_Schunck_flow(Ix, Iy, It, alpha=1.)
        flow_horn_schunck_bgr = flow_map_to_bgr(flow_horn_schunck)
        aae_horn_schunk = calculate_angular_error(flow_horn_schunck, groundtruth_flow)
        print('Average Angular error for Horn-Schunck Optical Flow: %.4f' %(aae_horn_schunk))
        
    #visualise the results
    if True:
        flow_bgr_gt = flow_map_to_bgr(groundtruth_flow)
        
        cv.imshow("GT", flow_bgr_gt)
        cv.waitKey()

        cv.imshow("Lucas-Kanade", flow_lucas_kanade_bgr)
        cv.waitKey()

        cv.imshow("Horn-Schunck", flow_horn_schunck_bgr)
        cv.waitKey()
        
        cv.destroyAllWindows()
