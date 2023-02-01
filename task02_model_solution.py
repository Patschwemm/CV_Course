import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as plt


def center_coordinates(data, n_views, n_features):
    centroids = np.mean(data, axis=1).reshape((-1, 1, 2))
    x = np.swapaxes(data.reshape((n_views, 2, n_features)), 1, 2)
    x_hat = x - centroids
    D = np.swapaxes(x_hat, 1, 2).reshape(n_views * 2, n_features)
    return D


def compute_s_m(D):
    U, W, V = la.svd(D)

    U = U[:, 0:3]
    V = V[0:3, :]
    W = np.diag(np.squeeze(W)[0:3])

    W_sqrt = np.sqrt(W)

    M = U @ W_sqrt

    S = W_sqrt @ V

    return M, S


def eliminate_ambiguity(S, M, n_views):
    A = np.zeros((3 * n_views, 6))
    B = np.zeros((3 * n_views, 1))

    for i in range(n_views):
        line_1 = i * 3
        line_2 = i * 3 + 1
        line_3 = i * 3 + 2

        mi1 = M[i * 2]
        mi2 = M[i * 2 + 1]

        m11, m12, m13 = mi1
        m21, m22, m23 = mi2

        A[line_1] = np.array([m11 ** 2,
                              2 * m11 * m12,
                              2 * m11 * m13,
                              m12 ** 2,
                              2 * m12 * m13,
                              m13 ** 2])
        A[line_2] = np.array([m21 ** 2,
                              2 * m21 * m22,
                              2 * m21 * m23,
                              m22 ** 2,
                              2 * m22 * m23,
                              m23 ** 2])
        A[line_3] = np.array([m11 * m21,
                              m12 * m21 + m11 * m22,
                              m13 * m21 + m11 * m23,
                              m12 * m22,
                              m13 * m22 + m12 * m23,
                              m13 * m23])

        B[line_1] = 1
        B[line_2] = 1
        B[line_3] = 0

    X = la.lstsq(A, B, rcond=-1)[0]

    l11, l12, l13, l22, l23, l33 = np.squeeze(X)

    L = np.array([
        [l11, l12, l13],
        [l12, l22, l23],
        [l13, l23, l33]
    ])

    C = la.cholesky(L)

    M_prime = M @ C
    S_prime = la.inv(C) @ S

    return M_prime, S_prime


def plot_reconstruction(data, M_prime, S_prime):
    centroids = np.mean(data, axis=1).reshape((-1, 2))
    data = np.swapaxes(data.reshape((n_views, 2, n_features)), 1, 2)
    pts3d = S_prime.T

    Ps = M_prime.reshape((n_views, 2, 3))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    for frame in range(0, n_views):
        ax.clear()

        im = cv2.imread('data/frame%08d.jpg' % (frame + 1), 0)
        ax.imshow(im, cmap='gray')

        features = data[frame]
        ax.scatter(features[:, 0], features[:, 1], color='blue')

        P2d = []
        for feature in range(n_features):
            p3d = pts3d[feature]
            P = Ps[frame]
            p2d = P @ p3d + centroids[frame]
            P2d.append(p2d)
        P2d = np.array(P2d)
        ax.scatter(P2d[:, 0], P2d[:, 1], color='red', alpha=0.5)

        plt.pause(1 / 50)
    plt.close(fig)


def plot_3d_points(pts3d, title):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])
    minv = np.min(pts3d) * 1.5
    maxv = np.max(pts3d) * 1.5
    ax.set_xlim([minv, maxv])
    ax.set_ylim([minv, maxv])
    ax.set_zlim([minv, maxv])

    plt.show()


if __name__ == '__main__':
    n_views = 101
    n_features = 215
    data = np.loadtxt('data/data_matrix.txt')
    D = center_coordinates(data, n_views=n_views, n_features=n_features)

    M, S = compute_s_m(D)
    M_prime, S_prime = eliminate_ambiguity(S, M, n_views)

    plot_reconstruction(data, M_prime, S_prime)

    # ======================================

    pts3d = S.T
    plot_3d_points(pts3d, title="with affine ambiguity")

    pts3d = S_prime.T
    plot_3d_points(pts3d, title="without affine ambiguity")

