import numpy as np
import cv2
import threading

def roberts(image, threshold):
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
            if roberts[i, j] > threshold:
                roberts[i, j] = 255
            else:
                roberts[i, j] = 0

    return roberts


def prewitt(image, threshold):
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
            if prewitt[i, j] > threshold:
                prewitt[i, j] = 255
            else:
                prewitt[i, j] = 0

    return prewitt


def sobel(image, threshold):
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
            if sobel[i, j] > threshold:
                sobel[i, j] = 255
            else:
                sobel[i, j] = 0

    return sobel

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    image1 = cv2.imread('C:\workspaces/PhotogrammetryWorkspace/test1.jpg', 0)
    image2 = cv2.imread('C:\workspaces/PhotogrammetryWorkspace/test2.jpg', 0)
    image3 = cv2.imread('C:\workspaces/PhotogrammetryWorkspace/test3.jpg', 0)

    image1 = cv2.resize(image1, (720,720))
    image2 = cv2.resize(image2, (720, 720))
    image3 = cv2.resize(image3, (720, 720))

    threshold = 200

    roberts_image1 = roberts(image1, threshold)
    roberts_image2 = roberts(image2, threshold)
    roberts_image3 = roberts(image3, threshold)

    prewitt_image1 = prewitt(image1, threshold)
    prewitt_image2 = prewitt(image2, threshold)
    prewitt_image3 = prewitt(image3, threshold)

    sobel_image1 = sobel(image1, threshold)
    sobel_image2 = sobel(image2, threshold)
    sobel_image3 = sobel(image3, threshold)

    cv2.imwrite('Test_Image_1.jpg', image1)
    cv2.imwrite('Test_Image_2.jpg', image2)
    cv2.imwrite('Test_Image_3.jpg', image3)

    cv2.imwrite('Roberts_Edges_Test_1_T200.jpg', roberts_image1)
    cv2.imwrite('Roberts_Edges_Test_2_T200.jpg', roberts_image2)
    cv2.imwrite('Roberts_Edges_Test_3_T200.jpg', roberts_image3)

    cv2.imwrite('Prewitt_Edges_Test_1_T200.jpg', prewitt_image1)
    cv2.imwrite('Prewitt_Edges_Test_2_T200.jpg', prewitt_image2)
    cv2.imwrite('Prewitt_Edges_Test_3_T200.jpg', prewitt_image3)

    cv2.imwrite('Sobel_Edges_Test_1_T200.jpg', sobel_image1)
    cv2.imwrite('Sobel_Edges_Test_2_T200.jpg', sobel_image2)
    cv2.imwrite('Sobel_Edges_Test_3_T200.jpg', sobel_image3)

    print('COMPLETED')