import numpy as np
import cv2
import matplotlib.pyplot as plt

n_views = 101
n_features = 215


data = np.loadtxt('data/data_matrix.txt')

centroids = np.mean(data, axis=1).reshape((-1, 2))
data = np.swapaxes(data.reshape((n_views, 2, n_features)), 1, 2)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)

for frame in range(0, n_views):
    ax.clear()

    im = cv2.imread('data/frame%08d.jpg' % (frame + 1), 0)
    ax.imshow(im, cmap='gray')

    features = data[frame]
    ax.scatter(features[:, 0], features[:, 1], color='blue')

    centroid_x, centroid_y = centroids[frame]
    ax.scatter(centroid_x, centroid_y, color='red', s=100)

    plt.pause(1/50)
