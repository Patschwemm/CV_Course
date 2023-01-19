import numpy as np
import cv2

# Load data
query_img = cv2.imread('data/1.jpg')
train_img = cv2.imread('data/2.jpg')


# Extract SIFT key points and features


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