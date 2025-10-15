import numpy as np
from copy import deepcopy
from .constants import IRIS_COLOR, PARAMETERS
from .image_processing import *

from skimage import filters, color
from PIL import Image

#find approximate position of pupil
def calculate_pupil_position(gray_noiseless: np.ndarray, intensity_threshold: float):
    img_height, img_width = gray_noiseless.shape
    group_radius = PARAMETERS.group_radius_factor * img_width

    #two point groups for two eyes case - for basic clustering
    group_a = []
    group_b = []

    #performs this again with a higher threshold in case it does not find anything
    while len(group_a) + len(group_b) == 0:
        bin_threshold_img = gray_noiseless < intensity_threshold
        dilated_img = dilation_preprocess(bin_threshold_img)
        eroded_img = erosion_preprocess(dilated_img)
        rows, cols = np.where(eroded_img == 1)

        for point in zip(rows, cols):
            if len(group_a) == 0:
                group_a.append(point)
            else:
                group_a_displacement = np.linalg.norm(np.average(group_a) - p)

                if group_a_displacement < group_radius:
                    group_a.append(point)
                else:
                    group_b.append(point)

        intensity_threshold += PARAMETERS.threshold_increase
    
    # set eye ball radii
    rad_x = int(PARAMETERS.eye_radius_factor * img_height) 
    rad_y = int(PARAMETERS.eye_radius_factor * img_width) 

    # get average intensities of each pixel group
    avg_a = np.average([gray_noiseless[x, y] for x, y in group_a])
    avg_b = np.average([gray_noiseless[x, y] for x, y in group_b])

    # pick the average coords based on the greater average intensity between the two groups
    # TODO this was actually lesser than before - check if it does worse the other way now
    if avg_a > avg_b:
        xavg, yavg = int(np.floor(np.average(group_a, 0)))
    else:
        xavg, yavg = int(np.floor(np.average(group_b, 0)))

    # get image coord slices
    top= np.max(xavg - rad_x, 0)
    bottom = np.min(xavg + rad_x, img_height)
    left = np.max(yavg - rad_y, 0) 
    right = np.min(yavg + rad_y, img_width)

    return top, bottom, left, right, xavg, yavg, rad_x, rad_y     
    

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


    coords  = calculate_pupil_position(gray_noiseless, intensity_threshold)
    top, bottom, left, right, xavg, yavg, rad_x, rad_y = coords
    
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