from enum import Enum

class IRIS_COLOR(Enum):
    GRAY = 0
    BLUE = 1
    GREEN = 2
    AMBER = 3
    BROWN = 4
    HAZEL = 5

class PARAMETERS:
    base_threshold = 0.05
    threshold_increase = 0.05
    num_binary_dilations = 2
    radius_max = 10
    radius_factor = 2.5