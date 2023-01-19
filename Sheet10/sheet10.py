import numpy as np
import cv2 as cv

# Load data
query_img = cv.imread('Sheet10/data/1.jpg')
train_img = cv.imread('Sheet10/data/2.jpg')


print(train_img.shape)

# Extract SIFT key points and features
sift = cv.SIFT_create()
# Extract SIFT key points and features
kp1, des1 = sift.detectAndCompute(query_img, None)
kp2, des2 = sift.detectAndCompute(train_img, None)

# Compute matches
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
ratio = 0.7
good_points = []
for m,n in matches:
    distance_ratio = m.distance / n.distance
    if distance_ratio > ratio :
        good_points.append([m, distance_ratio])
 
query_pts = np.float32([kp1[m[0].queryIdx].pt for m in good_points]).reshape(-1, 1, 2) 
 
train_pts = np.float32([kp2[m[0].trainIdx].pt for m in good_points]).reshape(-1, 1, 2) 

h, mask = cv.findHomography(query_pts, train_pts)

train_warp = cv.warpPerspective(train_img, h, (query_img.shape[1], query_img.shape[0]))
 

 
# sort matches to get 
matches = np.array(good_points)
idx = np.argsort(-matches[:, 1])
matches = matches[idx]

src_pts = []
dst_pts = []
for i, m in enumerate(matches):
    # only take first value because second value is the distance
    src_pts.append(kp1[m[0].queryIdx].pt)
    dst_pts.append(kp2[m[0].trainIdx].pt)
    # need eight points only for homography
    if i == 4:
        break
    
A = []
for s, d in zip(src_pts, dst_pts):
    # maybe x and y need to be swapped
    #s = xy, d = x'y'
    A.append([s[0], s[1], 1, 0, 0, 0, -d[0]*s[0],  -d[0]*s[1], -d[0]])
    A.append([0, 0, 0, s[0], s[1], 1, -d[1]*s[0],  -d[1]*s[1], -d[1]])
A = np.array(A)

U, D, V = np.linalg.svd(A)
eigvals_idx = np.argsort(D)
# take lowest eigval for the eigenvector that is our homography
V = V[eigvals_idx]
H = V[0].reshape(3,3)

# normalize H to sum up to one
H = H * ( 1 / (H**2).sum())
print(H)

height, width, _ = query_img.shape
height = np.arange(0, height)
width = np.arange(0, width)
# make grid of coordinates
grid = np.meshgrid(width, height)
ones = np.ones_like(grid[0][:, :, None])
grid = np.concatenate((grid[1][:,:,None],grid[0][:,:,None], ones), axis=-1)

# p_prime = grid * H

train_warp = cv.warpPerspective(train_img, H, (query_img.shape[1], query_img.shape[0]))


# _, ax = plt.subplots(1, 1)
train_warp = cv.cvtColor(train_warp, cv.COLOR_BGR2RGB)
# ax.imshow(train_warp)
cv.imshow("img", train_warp)
cv.waitkey()




# Compute matches


# Projection matrixs for query_img and train_img
P_q = np.array([[1.0, 0, 0, 0],
                [0, 1.0, 0, 0],
                [0, 0, 1.0, 0]])

P_t = np.array([[1.0, 0, 0, 1],
                [0, 1.0, 0, 1],
                [0, 0, 1.0, 0]])

# Compute 3D points


# Visualization