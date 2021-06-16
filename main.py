import numpy as np
import cv2

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

    m, n = image.shape # get dimensions of image
    roberts = np.zeros((m, n)) # allocates space for the new image array

    for i in range(1, m): # loops through image x-axis position(i, j) is the bottom right hand coner of the mask
        for j in range(1, n):
            img = image[i - 1:i + 1, j - 1:j + 1] # saves the values of the respective pixels underneath the mask

            x = roberts_x * img # multiply the roberts mask values by their respective pixel values
            r_x = np.square(x.sum()) # squares the sum of the mask response

            y = roberts_y * img # process is repeated for the second roberts mask
            r_y = np.square(y.sum())
            # the roberts response is the sqrt of the square of the sum of the pixel responses
            roberts[i, j] = np.sqrt(r_x + r_y)

    for i in range(m): # applies the pixel response threshold to the resulting image.
        for j in range(n):
            if roberts[i, j] > threshold:  # if the response is above the threshold then the value is changed to 255
                roberts[i, j] = 255
            else: # if the response is below the threshold then the value is changed to 0
                roberts[i, j] = 0

    return roberts # returns the new image of the response

# similar process is applied here than for the roberts operator, but just with a different mask
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

# similar process is applied here than for the roberts operator, but just with a different mask
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

    image = cv2.imread('C:\workspaces/PhotogrammetryWorkspace/test1.jpg', 0) #opening of image using opencv

    image = cv2.resize(image, (720,720)) #resizing image using opencv in order to speed up processing

    threshold = 127 #edge detection threshold

    # function calls
    roberts_image = roberts(image, threshold)
    prewitt_image = prewitt(image, threshold)
    sobel_image = sobel(image, threshold)

    # display and save original image (post resizing)
    cv2.imshow('Original Test Image', image)
    cv2.imwrite('Test_Image_1.jpg', image)

    # show edge detection results
    cv2.imshow('Roberts_Edges_Test_T' + str(threshold), roberts_image)
    cv2.imshow('Prewitt_Edges_Test_T' + str(threshold), prewitt_image)
    cv2.imshow('Sobel_Edges_Test_T' + str(threshold), sobel_image)

    # write edge detection results to file
    cv2.imwrite('Roberts_Edges_Test_T' + str(threshold) + '.jpg', roberts_image)
    cv2.imwrite('Prewitt_Edges_Test_T' + str(threshold) + '.jpg', prewitt_image)
    cv2.imwrite('Sobel_Edges_Test_T' + str(threshold) + '.jpg', sobel_image)


    print('COMPLETED')

    # shows edge detection results until keyboard input
    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows()