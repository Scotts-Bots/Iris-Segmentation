import numpy as np
from copy import deepcopy
from .constants import IRIS_COLOR, PARAMETERS

from scipy.ndimage import morphology
from skimage import io
from skimage.transform import hough_circle, hough_circle_peaks
from skimage import filters, feature, morphology
from PIL import Image


def predict_eye_color(rgb_val: tuple[int, int, int]) -> IRIS_COLOR:
    '''
        predicts one of the six common iris colours from the average intensity of the iris.
    '''

    red, green, blue = rgb_val

    if (
        np.abs(red - green) < 15
        and np.abs(green - blue) < 15
        and np.abs(red - blue) < 15
        and red > 100 and green > 100 and blue > 100
    ):
        return IRIS_COLOR.GRAY
    
    # blue component is greatest
    elif blue > green and blue > red:
        return IRIS_COLOR.BLUE
    
    # green component is greatest
    elif green > red and green > blue:
        return IRIS_COLOR.GREEN
    
    # red component is greatest
    else: 
        if blue < 10 and red > 150:
            return IRIS_COLOR.AMBER
        elif red < 100 or blue < 60 and green < 80:
            return IRIS_COLOR.BROWN
        else:
            return IRIS_COLOR.HAZEL

#preprocesses an image by performing dilation and removing unnecessary noise in the image
def dilation_preprocess(image: np.ndarray) -> np.ndarray:
    img_width, img_height = image.shape
    dilated_img = deepcopy(image)

    for _ in range(PARAMETERS.num_binary_dilations):
        dilated_img = morphology.binary_dilation(dilated_img)

    coords_dil = np.where(dilated_img == 1)
    average = np.average(coords_dil,1)

    # calculate center pixel and distance from center to the average position of all foreground pixels
    center = np.array([img_width/2, img_height/2])
    radius_x, radius_y = np.abs(center - average)

    # TODO Try to change this to be a factor of the image OR is this even necessary??
    if radius_y < PARAMETERS.radius_max: 
        radius_y = 0.2*img_height
    if radius_x < PARAMETERS.radius_max: 
        radius_x = 0.2*img_width

    # calculate a point we want to focus on
    turning_point = (center + average) / 2

    # remove noise on the sides of the image
    X, Y = np.ogrid[:img_width, :img_height]
    mask_x = ((X < turning_point[0] - PARAMETERS.radius_factor * radius_x) 
              | (X > turning_point[0] + PARAMETERS.radius_factor * radius_x))
    mask_y = ((Y < turning_point[1] - PARAMETERS.radius_factor * radius_y) 
              | (Y > turning_point[1] + PARAMETERS.radius_factor * radius_y))
    dilated_img[mask_x, :] = 0
    dilated_img[:, mask_y] = 0

    return dilated_img

#perform circle hit-or-miss on a thresholded image to find pupil
def erosion_preprocess(image: np.ndarray):
    disk = morphology.disk(PARAMETERS.erosion_disk_radius)
    eroded = deepcopy(image)

    while np.count_nonzero(eroded) > PARAMETERS.erosion_max_pixels:
        eroded = morphology.binary_erosion(eroded, disk)

    return np.where(eroded == 1)
   
#find approximate position of pupil
def calculate_pupil_position(gray_noiseless, intensity_threshold):
    group_a = []
    group_b = []
    num_coords = 0

    #performs this again with a higher threshold in case it does not find anything
    while num_coords == 0:
        bin_threshold_img = gray_noiseless < intensity_threshold
        dilated_img = dilation_preprocess(bin_threshold_img)
        coords = erosion_preprocess(dilated_img)

        for r in range(len(coords[0])):
            p = np.array([coords[0][r],coords[1][r]])
            if len(group_a) == 0:
                group_a.append(p)
            elif np.linalg.norm(np.average(group_a)-p) >= 0.25*test_img.shape[1]:
                group_b.append(p)
            else:
                group_a.append(p)

        num_coords = len(group_a)+len(group_b)
        intensity_threshold += PARAMETERS.threshold_increase
           
    radx = int(0.12*test_img.shape[0])
    rady = int(0.12*test_img.shape[1])

    a_avg_intensity = 0
    for acoord in group_a:
        a_avg_intensity += gray_noiseless[acoord[0]][acoord[1]]

    b_avg_intensity = 0
    for bcoord in group_b:
        b_avg_intensity += gray_noiseless[bcoord[0]][bcoord[1]]

    if len(group_a) == 0:
        avg_a = 2000
    else:
        avg_a = a_avg_intensity/len(group_a)
    if len(group_b) == 0:
        avg_b = 2000
    else:
        avg_b = b_avg_intensity/len(group_b)

    #xavg = int(np.floor(np.average(coords[0])))
    #yavg = int(np.floor(np.average(coords[1])))
    #print(len(group_a),len(group_b))
    if avg_b >avg_a:
        xavg,yavg = np.average(group_a,0)
    else:
        xavg,yavg = np.average(group_b,0)
    xavg = int(np.floor(xavg))
    yavg = int(np.floor(yavg))

    #print(xavg,yavg,radx,rady)

    coord1,coord2,coord3,coord4 = xavg-radx,xavg+radx,yavg-rady,yavg+rady
    if coord1 < 0: coord1 = 0
    if coord2 >= test_img.shape[0]: coord2 = test_img.shape[0]-1
    if coord3 < 0: coord3 = 0
    if coord4 >= test_img.shape[1]: coord4 = test_img.shape[1]-1

    return coord1,coord2,coord3,coord4,xavg,yavg,radx,rady     
    



