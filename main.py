from refactor.iris_segment import *

if __name__ == "__main__":
    predict_eye_color, _ = iris_segmentation_and_prediction("images/amber/amber4.jpg")
    print(f'prediction: {predict_eye_color}')