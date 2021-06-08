import numpy as np
import cv2


def roberts(image):
    # Roberts mask
    roberts_x = np.array([
        [-1, 0],
        [0, 1]
    ])
    roberts_y = np.array([
        [0, -1],
        [1, 0]
    ])

    m, n = image.shape
    roberts = np.zeros((m, n))

    for i in range(2, m):
        for j in range(2, n):
            img = image[i - 1:i + 1, j - 1:j + 1]
            x = roberts_x * img
            r_x = np.square(x.sum())
            y = roberts_y * img
            r_y = np.square(y.sum())
            roberts[i, j] = np.sqrt(r_x + r_y)

    for i in range(m):
        for j in range(n):
            if roberts[i, j] > 127:
                roberts[i, j] = 255
            else:
                roberts[i, j] = 0
    return roberts


def prewitt(image):
    # Prewitt mask
    prewitt_x = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]])
    prewitt_y = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]])

    m, n = image.shape
    prewitt = np.zeros((m, n))

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            img = image[i-1:i+2, j-1:j+2]
            x = prewitt_x*img
            r_x = np.square(x.sum())
            y = prewitt_y*img
            r_y = np.square(y.sum())
            prewitt[i, j] = np.sqrt(r_x + r_y)

    for i in range(m):
        for j in range(n):
            if prewitt[i, j] > 127:
                prewitt[i, j] = 255
            else:
                prewitt[i, j] = 0
    return prewitt


def sobel(image):
    # Sobel mask
    sobel_x = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]])
    sobel_y = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]])

    m, n = image.shape
    sobel = np.zeros((m, n))

    for i in range(2, m - 1):
        for j in range(2, n - 1):
            img=image[i - 1:i + 2, j - 1:j + 2]
            x = sobel_x * img
            r_x = np.square(x.sum())
            y = sobel_y * img
            r_y = np.square(y.sum())
            sobel[i, j] = np.sqrt(r_x + r_y)

    for i in range(m):
        for j in range(n):
            if sobel[i, j] > 127:
                sobel[i, j] = 255
            else:
                sobel[i, j] = 0
    return sobel


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #image = cv2.imread('C:\workspaces/PhotogrammetryWorkspace/test1.jpg', 0)
    #image = cv2.imread('C:\workspaces/PhotogrammetryWorkspace/test2.jpg', 0)
    image = cv2.imread('C:\workspaces/PhotogrammetryWorkspace/test3.jpg', 0)

    image = cv2.resize(image, (720,720))
    cv2.imshow('Original Image', image)

    roberts_image = roberts(image)
    prewitt_image = prewitt(image)
    sobel_image = sobel(image)

    cv2.imshow('Roberts Edges', roberts_image)
    cv2.imshow('Prewitt Edges', prewitt_image)
    cv2.imshow('Sobel Edges', sobel_image)

    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows()
