from configuration.config import anchors_path, Input_shape, dataset_class_file, train_file_path
from utils.yolo_utils import read_anchors, read_classes, letterbox_image, preprocess_true_boxes

import numpy as np
from PIL import Image
import os



def get_training_data(annotation_path, data_path, input_shape, anchors, num_classes, max_boxes=1):
    """
    processes the data into standard shape
    :param annotation_path: path_to_image box1,box2,...,boxN with boxX: x_min,y_min,x_max,y_max,class_index
    :param data_path: saver at "/home/minh/stage/train.npz"
    :param input_shape: (416, 416)
    :param max_boxes: 100: maximum number objects of an image
    :param load_previous: for 2nd, 3th, .. using
    :return: image_data [N, 416, 416, 3] not yet normalized, N: number of image
             box_data: box format: [N, 100, 5], 100: maximum number of an image
                                                5: top_left{x_min,y_min},bottom_right{x_max,y_max},class_index (no space)
                                                /home/minh/keras-yolo3/VOCdevkit/VOC2007/JPEGImages/000012.jpg 156,97,351,270,6
    """
    image_data = []
    box_data = []
    image_shape = []
    with open(annotation_path) as f:
        GG = f.readlines()
        np.random.shuffle(GG)
        for line in (GG):
            line = line.split('$$')
            filename = line[0]
            if filename[-1] == '\n':
                filename = filename[:-1]
            image = Image.open(filename)
            # shape_image = [360 640], boxed_image = [416 416]
            boxed_image, shape_image = letterbox_image(image, tuple(reversed(input_shape)))
            image_data.append(np.array(boxed_image, dtype=np.uint8))  # pixel: [0:255] uint8:[-128, 127]
            image_shape.append(np.array(shape_image))

            boxes = np.zeros((max_boxes, 5), dtype=np.int32)
            # correct the BBs to the image resize
            if len(line)==1:  # if there is no object in this image
                box_data.append(boxes)
            for i, box in enumerate(line[1:]):
                # take first n boxes into account. Here n = max_boxes.
                if i < max_boxes:
                    boxes[i] = np.array(list(map(int, box.split(','))))
                else:
                    break
                # image_size = [640 360]
                image_size = np.array(image.size)
                # Reverse the input_shape array. input_size = [416 416]
                input_size = np.array(input_shape[::-1])
                # new_size = [416 234]. It is the image after resizing.
                new_size = (image_size * np.min(input_size*1.0/image_size)).astype(np.int32)
                # Correct BB to new image. As the image resizes, the BBs should also resize accordingly.
                boxes[i:i+1, 0:2] = (boxes[i:i+1, 0:2]*new_size*1.0/image_size + (input_size-new_size)/2.).astype(np.int32)
                boxes[i:i+1, 2:4] = (boxes[i:i+1, 2:4]*new_size*1.0/image_size + (input_size-new_size)/2.).astype(np.int32)
            box_data.append(boxes)
    image_shape = np.array(image_shape)
    image_data = np.array(image_data)
    box_data = (np.array(box_data))
    y_true = preprocess_true_boxes(box_data, input_shape[0], anchors, num_classes)
    # # np.savez(data_path, image_data=image_data, box_data=box_data, image_shape=image_shape)
    # np.savez(data_path, image_data=image_data, box_data=box_data, image_shape=image_shape, y_true0=y_true[0], y_true1=y_true[1], y_true2=y_true[2])
    # # np.savez(data_path, image_data=image_data, box_data=box_data, image_shape=image_shape, y_true=y_true)
    # print('Saving training data into ' + data_path)
    # return image_data, box_data, image_shape

    #  image_data = [N, 416,416,color_channels], box_data = [N,max_boxes, 5], image_shape = [N,2]
    return image_data, box_data, image_shape, y_true


max_boxes = 1
classes_data = read_classes(dataset_class_file)
anchors = read_anchors(anchors_path)
annotation_path = train_file_path + '/train_batch_0.txt'
input_shape = [Input_shape, Input_shape]
#print("\n classes in dataset \n")
#print(classes_data)
#print("\n anchors that we are using \n")
#print(anchors)

image_data, box_data, image_shape, y_true = get_training_data(annotation_path, '', input_shape, anchors, len(classes_data), max_boxes=max_boxes)
#print("\n box_data shape : {} \n".format(box_data.shape))
'''
for j in range(32):
    print("Now looking at box_data for image {} in the batch".format(j))
    for i in range(max_boxes):
        print(box_data[j][i])
'''
'''
for i in range(13):
    for j in range(13):
        print("{} , {} in the grid cell".format(j,i))
        for k in range(3):
            print("filter {}".format(k))
            print(y_true[2][0][j][i][k])
'''

for i in range(3):
    print(y_true[i].shape)
