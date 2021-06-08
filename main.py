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

    for i in range(2, m - 1):
        for j in range(2, n - 1):
            x = np.square(roberts_x[0, 0] * image[i - 1, j - 1] + roberts_x[0, 1] * image[i - 1, j] + roberts_x[1, 0] * image[i, j - 1] + roberts_x[1, 1] * image[i, j])
            y = np.square( roberts_y[0, 0] * image[i - 1, j - 1] + roberts_y[0, 1] * image[i - 1, j] + roberts_y[1, 0] * image[i, j - 1] + roberts_y[1, 1] * image[i, j])
            roberts[i,j] = np.sqrt(x+y)

    for i in range(m):
        for j in range(n):
            if roberts[i,j]>50: #was 127
                roberts[i,j]=255
            else:
                roberts[i,j]=0
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

    for i in range(2, m - 1):
        for j in range(2, n - 1):
            roberts[i, j] = prewitt_x[0, 0] * image[i - 1, j - 1] + prewitt_x[0, 1] * image[i - 1, j] + prewitt_x[
                0, 2] * image[i - 1, j + 1] + prewitt_x[1, 0] * image[i, j - 1] + prewitt_x[1, 1] * image[i, j] + \
                            prewitt_x[1, 2] * image[i, j + 1] + prewitt_x[2, 0] * image[i + 1, j - 1] + prewitt_x[
                                2, 1] * image[i + 1, j] + prewitt_x[2, 2] * image[i + 1, j + 1]

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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    image = cv2.imread('C:\workspaces/PhotogrammetryWorkspace/test1.jpg', 0)
    image = cv2.resize(image, (800, 800))
    cv2.imshow('Original Image', image)

    roberts_image = roberts(image)
    #prewitt_image = prewitt(image)
    #sobel_image = sobel(image)

    image = cv2.resize(image, (800, 600))
    roberts_sharpened = cv2.resize(roberts_image, (800, 600))
    cv2.imshow('Roberts Sharpened', roberts_sharpened)

    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows()
