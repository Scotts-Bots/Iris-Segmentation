import numpy as np
from copy import deepcopy
from .constants import IRIS_COLOR, PARAMETERS

from skimage import filters, color
from PIL import Image

def iris_segmentation_and_prediction(image_path: str) -> IRIS_COLOR:
    '''
        Segments an iris from an image of an eye and predicts the color.

        Args:
            image_path: relative path of image of an eye/eyes

        Returns:
            (predicted color, image of segmented iris)
    '''

    #read in input image and convert to gray image then apply smoothing
    img = np.array(Image.open(image_path))
    gray_img = color.rgb2gray(img)
    gray_noiseless = filters.gaussian(gray_img, 1)

    #calculate a threshold intensity
    min_intensity = np.min(gray_noiseless)
    if min_intensity < 0.1:
        intensity_threshold = PARAMETERS.base_threshold + 2*min_intensity
    else:
        intensity_threshold = PARAMETERS.base_threshold + min_intensity


    coord1,coord2,coord3,coord4,xavg,yavg,radx,rady = calculate_pupil_position(img,gray_noiseless,intensity_threshold)
    
    #after getting a sliced image of the pupil
    eye_img = img[coord1:coord2,coord3:coord4]
    gray_eye = gray_img[coord1:coord2,coord3:coord4]
    #eye_threshold = gray_threshold[xavg-100:xavg+100,yavg-100:yavg+100]
    eye_noiseless = gray_noiseless[coord1:coord2,coord3:coord4]
    eye_threshold = np.zeros(gray_eye.shape)
    for i in range(gray_eye.shape[0]):
        for j in range(gray_eye.shape[1]):
            if eye_noiseless[i][j] > intensity_threshold:
                eye_threshold[i][j] = 1

    #get center and radius of circle from canny edge and hough circle detection
    cx,cy,radii = get_iris_center(eye_threshold,eye_img)

    greatest_circle = 0
    cxt = cy[greatest_circle] + xavg - radx
    cyt = cx[greatest_circle] + yavg - rady

    radi = radii[greatest_circle]*(img.shape[1]/eye_img.shape[1])

    iris_img = deepcopy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.linalg.norm(np.array([i,j])-np.array([cxt,cyt])) >= 0.6*radi:
                iris_img[i][j] = [0,0,0]


    gray_iris = color.rgb2gray(iris_img)
    #plt.imshow(gray_iris,cmap=plt.cm.gray)

    eye_threshold = deepcopy(gray_iris)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if gray_iris[i][j] < 0.12 or gray_iris[i][j] > 0.7:
                eye_threshold[i][j] = 0
            else:
                eye_threshold[i][j] = 1

    #calculate the average colour from the segmented iris
    avg_colour = np.array([0,0,0])
    count = 0
    output_img = deepcopy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (eye_threshold[i][j]):
                avg_colour += img[i][j]
                count += 1
            else:
                output_img[i][j] = [0,0,0]
    
    avg_colour = avg_colour/count
    avg_colour[np.isnan(avg_colour)] = 0
    colour = np.zeros((3,3,3))
    for i in range(len(avg_colour)):
        avg_colour[i] = avg_colour[i]/255
    colour[1][1] = avg_colour

    reqColour = []
    for i in range(3):
        reqColour.append(int(np.floor(avg_colour[i]*256)))

    #predict colour from average intensity and return result
    out_colour = predict_eye_color(reqColour)
    #print(out_colour) <---- IF YOU WANT TO PRINT OUT THE COLOUR WHEN RUNNING THROUGH ALL THE IMAGES
    return out_colour, output_img