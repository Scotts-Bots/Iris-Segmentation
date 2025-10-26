from refactor.iris_segment import *

if __name__ == "__main__":
    # img_path = "images/amber/amber3.jpg"
    # predict_eye_color, _ = iris_segmentation_and_prediction(img_path)
    # print(f'prediction: {predict_eye_color}')

    #This is the details of the image data set needed if you want to run the below code
    num_colours = 6
    colours = ['amber', 'blue', 'brown', 'gray', 'green', 'hazel']
    num_images = [11,11,20,18,18,18]
    num_correct_predictions = [0,0,0,0,0,0]
    results = []

    #This calls the main function for all images in the dataset and returns the output segmented iris image as well as the predicted colour
    for c in range(len(colours)):
        temp = []
        for j in range(1, num_images[c] + 1):
            img_path = f'images/{colours[c]}/{colours[c]}{str(j)}.jpg'
            predict_eye_color, _ = iris_segmentation_and_prediction(img_path)
            temp.append(predict_eye_color)
            if predict_eye_color == colours[c]:
                num_correct_predictions[c] += 1
        results.append(temp)

    for j in range(6):
        print(colours[j],"\ttotal number images: ", num_images[j], "\ttotal predicted correct: ", num_correct_predictions[j],"\tsuccess rate: ", (num_correct_predictions[j]/num_images[j])*100)

    totalcorrect = np.sum(num_correct_predictions)
    total = np.sum(num_images)
    print("total success rate: ",totalcorrect/total)