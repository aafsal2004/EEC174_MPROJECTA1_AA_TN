import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import glob


#Paths can't be hardcoded, so they must be inputted through the command line
#e.g python yolo_img_detector.py --weights yolo_files/yolov3.weights
parser = argparse.ArgumentParser(description= "Used to take in Yolo Weights, Config, Labels, and image path")
parser.add_argument('--image_path', required=True, nargs='+', help="Path to images/image")
parser.add_argument('--weights', required=True, help='Path to Yolo Weights')
parser.add_argument('--cfg', required=True, help='Path to Yolo Config')
parser.add_argument('--labels', required=True ,help='Path to Classifier Labels')

args = parser.parse_args()

#We need to open the label names folder and add them to a list
with open(args.labels) as f:
    #appending labels list
    #.strip() gets rid of the '\n' character at the end of each line
    labels = [line.strip() for line in f]

#We need to determine whether or not the image path is one image, a list of images, or a folder directory
image_paths = []
for path in args.image_path :
    if os.path.isdir(path):
        #checks if the path passed in was a folder
        image_paths.extend(sorted(glob.glob(os.path.join(path, '*.jpg'))))
    elif os.path.isfile(path):
        #checks if path is just a single path
        image_paths.append(path)
    else :
        print(f"{path} does not exist")

#Ripped from tutorial
network = cv2.dnn.readNetFromDarknet(args.cfg, args.weights)
layers = network.getLayerNames()
yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']

for images in image_paths:
    image = cv2.imread(images)
    input_blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    network.setInput(input_blob)
    output = network.forward(yolo_layers)

    print(type(output))  # <class 'list'>
    print(type(output[0]))  # <class 'numpy.ndarray'>

    bounding_boxes = []
    confidences = []
    classes = []
    probability_minimum = 0.5
    threshold = 0.3
    h, w = image.shape[:2]

    for result in output:
        for detection in result:
            scores = detection[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current > probability_minimum:
                box_current = detection[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current.astype('int')
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                classes.append(class_current)

    
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
    coco_labels = 80
    np.random.seed(42)
    colours = np.random.randint(0, 255, size=(coco_labels, 3), dtype='uint8')
    
    if len(results) > 0:
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            colour_box = [int(j) for j in colours[classes[i]]]
            #Fetching the label for its respective box
            current_label = labels[classes[i]]
            cv2.rectangle(image, (x_min, y_min), (x_min + box_width, y_min + box_height),
                        colour_box, 5)
            text_box = '{}({:.2f})'.format(current_label,confidences[i])
            cv2.putText(image, text_box, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colour_box, 5)

    plt.figure()
    plt.rcParams['figure.figsize'] = (10.0, 10.0)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.show()

    













