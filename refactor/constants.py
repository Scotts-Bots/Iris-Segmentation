from enum import Enum

class IRIS_COLOR(Enum):
    GRAY = "gray"
    BLUE = "blue"
    GREEN = "green"
    AMBER = "amber"
    BROWN = "brown"
    HAZEL = "hazel"

class PARAMETERS:
    base_threshold = 0.05
    threshold_increase = 0.05
    num_binary_dilations = 2
    radius_max = 10
    radius_factor = 2.5
    erosion_disk_radius = 2
    erosion_max_pixels = 500
    group_radius_factor = 0.25
    eye_radius_factor = 0.12
    iris_inner_radius = 0.3
    iris_outer_radius = 0.7