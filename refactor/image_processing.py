import numpy as np
from copy import deepcopy
from .constants import IRIS_COLOR, PARAMETERS

from scipy.ndimage import morphology
from skimage import morphology
import matplotlib.pyplot as plt


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
        return IRIS_COLOR.GRAY.value
    
    # blue component is greatest
    elif blue > green and blue > red:
        return IRIS_COLOR.BLUE.value
    
    # green component is greatest
    elif green > red and green > blue:
        return IRIS_COLOR.GREEN.value
    
    # red component is greatest
    else: 
        if blue < 10 and red > 150:
            return IRIS_COLOR.AMBER.value
        elif red < 100 or blue < 60 and green < 80:
            return IRIS_COLOR.BROWN.value
        else:
            return IRIS_COLOR.HAZEL.value

#preprocesses an image by performing dilation and removing unnecessary noise in the image
def dilation_preprocess(image: np.ndarray) -> np.ndarray:
    img_height, img_width = image.shape
    dilated_img = deepcopy(image)

    for _ in range(PARAMETERS.num_binary_dilations):
        dilated_img = morphology.binary_dilation(dilated_img)

    coords_dil = np.where(dilated_img == 1)
    average = np.average(coords_dil, 1)

    # calculate center pixel and distance from center to the average position of all foreground pixels
    center = np.array([img_height/2, img_width/2])
    radius_x, radius_y = np.abs(center - average)

    # TODO Try to change this to be a factor of the image OR is this even necessary??
    if radius_y < PARAMETERS.radius_max: 
        radius_y = 0.2*img_height
    if radius_x < PARAMETERS.radius_max: 
        radius_x = 0.2*img_width

    # calculate a point we want to focus on
    turning_point = (center + average) / 2

    # remove noise on the sides of the image
    Y, X = np.ogrid[:img_height, :img_width]
    mask_y = ((Y < turning_point[0] - PARAMETERS.radius_factor * radius_y) 
              | (Y > turning_point[0] + PARAMETERS.radius_factor * radius_y))
    mask_x = ((X < turning_point[1] - PARAMETERS.radius_factor * radius_x) 
              | (X > turning_point[1] + PARAMETERS.radius_factor * radius_x))
    
    dilated_img[mask_y.ravel(), :] = 0
    dilated_img[:, mask_x.ravel()] = 0

    plt.imsave("refactor/images_out/2_dilated.png", dilated_img, cmap=plt.cm.gray)

    return dilated_img

#perform circle hit-or-miss on a thresholded image to find pupil
def erosion_preprocess(image: np.ndarray) -> np.ndarray:
    disk = morphology.disk(PARAMETERS.erosion_disk_radius)
    eroded = deepcopy(image)

    while np.count_nonzero(eroded) > PARAMETERS.erosion_max_pixels:
        eroded = morphology.binary_erosion(eroded, disk)

    plt.imsave("refactor/images_out/3_eroded.png", eroded, cmap=plt.cm.gray)

    return eroded



