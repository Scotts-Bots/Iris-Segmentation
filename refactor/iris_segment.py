import numpy as np
from copy import deepcopy
from PIL import Image
from skimage import filters, color, feature
from skimage.transform import hough_circle, hough_circle_peaks
import matplotlib.pyplot as plt

from .constants import IRIS_COLOR, PARAMETERS
from .image_processing import *

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

    #calculate a threshold intensity and binary thresholded image
    min_intensity = np.min(gray_noiseless)

    if min_intensity < 0.1:
        intensity_threshold = PARAMETERS.base_threshold + 2*min_intensity
    else:
        intensity_threshold = PARAMETERS.base_threshold + min_intensity

    bin_threshold_img = gray_noiseless < intensity_threshold

    plt.imsave("refactor/images_out/1_bin_threshold.png", bin_threshold_img, cmap=plt.cm.gray)

    # get pupil position coords and sliced eye image
    coords  = calculate_pupil_position(gray_noiseless, intensity_threshold)
    top, bottom, left, right, yavg, xavg, rad_x, rad_y = coords
    clipped_bin_img = bin_threshold_img[top:bottom, left:right]

    plt.imsave("refactor/images_out/4_eye_img.png", clipped_bin_img, cmap=plt.cm.gray)

    # get center and radius of circle from canny edge and hough circle detection
    cx, cy, radii = get_iris_center(clipped_bin_img)

    cxt = cx[0] + xavg - rad_x
    cyt = cy[0] + yavg - rad_y
    radi = radii[0] * (img.shape[1] / clipped_bin_img.shape[1])

    # print(cx, cy, radii)

    # create a mask for pixels within the iris radius
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    dist_from_center = np.sqrt((X - cxt)**2 + (Y - cyt)**2)
    mask = ((dist_from_center > PARAMETERS.iris_inner_radius * radi)
            & (dist_from_center < PARAMETERS.iris_outer_radius * radi))

    iris_img = deepcopy(img)
    iris_img[~mask] = [0, 0, 0]
    # eye_threshold = bin_threshold_img[~mask] = 0
    
    plt.imsave("refactor/images_out/5_iris.png", iris_img, cmap=plt.cm.gray)

    avg_colour = np.floor(img[mask].mean(axis=0)).astype(int)
    # print(avg_colour)

    color_class = predict_eye_color(avg_colour)
    
    return color_class, iris_img

#find approximate position of pupil
def calculate_pupil_position(img: np.ndarray, intensity_threshold: float):
    img_height, img_width = img.shape
    group_radius = PARAMETERS.group_radius_factor * img_width

    #two point groups for two eyes case - for basic clustering
    group_a = []
    group_b = []

    #performs this again with a higher threshold in case it does not find anything
    while len(group_a) + len(group_b) == 0:
        threshold_img = img < intensity_threshold
        dilated_img = dilation_preprocess(threshold_img)
        eroded_img = erosion_preprocess(dilated_img)
        rows, cols = np.where(eroded_img == 1)

        for point in zip(rows, cols):
            if len(group_a) == 0:
                group_a.append(point)
            else:
                group_a_displacement = np.linalg.norm(np.average(group_a) - point)

                if group_a_displacement < group_radius:
                    group_a.append(point)
                else:
                    group_b.append(point)

        intensity_threshold += PARAMETERS.threshold_increase
    
    # set eye ball radii
    rad_y = int(PARAMETERS.eye_radius_factor * img_height) 
    rad_x = int(PARAMETERS.eye_radius_factor * img_width) 

    # get average intensities of each pixel group
    avg_a = np.average([img[x, y] for x, y in group_a])
    avg_b = np.average([img[x, y] for x, y in group_b])

    # pick the average coords based on the greater average intensity between the two groups
    # TODO this was actually lesser than before - check if it does worse the other way now
    if avg_a < avg_b or np.isnan(avg_b):
        yavg, xavg = np.floor(np.average(group_a, 0))
    else:
        yavg, xavg = np.floor(np.average(group_b, 0))
    yavg, xavg = int(yavg), int(xavg)

    # get image coord slices
    top= np.max([yavg - rad_y, 0])
    bottom = np.min([yavg + rad_y, img_height])
    left = np.max([xavg - rad_x, 0]) 
    right = np.min([xavg + rad_x, img_width])

    return top, bottom, left, right, yavg, xavg, rad_x, rad_y     
    
#find the iris center using canny edge detection and hough circles
def get_iris_center(thresholded_img: np.ndarray): 
    h = thresholded_img.shape[0]

    eye_edges = feature.canny(thresholded_img)

    hough_radii = np.arange(int(h/6), int(h/2), 1)
    hough_spaces = hough_circle(eye_edges, hough_radii)
    _, cx, cy, radii = hough_circle_peaks(hough_spaces, hough_radii, total_num_peaks=1)

    return cx, cy, radii

