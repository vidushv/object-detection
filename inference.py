# from argparse import ArgumentParser

from configuration.config import Input_shape, channels, threshold, ignore_thresh, visible_GPU, num_epochs, max_boxes
from configuration.config import anchors_path, project_path, dataset_name, dataset_class_file, model_checkpoint_path, font_file
from configuration.config import input_test_img_path, output_test_img_path, train_file_path, val_file_path

from model.net import YOLOv3
from model.detect_function import predict

from utils.yolo_utils import read_anchors, read_classes, letterbox_image, calc_iou  # , resize_image

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer  # to calculate FPS
from pathlib import Path
import numpy as np
import tensorflow as tf
import argparse
import colorsys
import random
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = visible_GPU

parser = argparse.ArgumentParser(description='COCO or VOC or customized dataset')
parser.add_argument('--COCO', action='store_true', help='COCO flag')
# parser.add_argument('--other', action='store_true', help='customized dataset flag')
parser.add_argument('--image', action='store_true', help='image detection')
parser.add_argument('--name', type=str, default='', help='name of the test image')
parser.add_argument('--train_batch', type=int, default=-1, help='inference on one train batch')
parser.add_argument('--val_batch', type=int, default=-1, help='inference on one validation batch')
args = parser.parse_args()


class YOLO(object):
    def __init__(self):

        self.anchors_path = anchors_path
        self.COCO = False
        self.trainable = True
        # self.args = self.argument()
        if args.COCO:
            print("-----------COCO-----------")
            self.COCO = True
            self.trainable = False
            self.class_names = read_classes(project_path+'/data/coco_classes.txt')
        else:
            print("----------{}-----------".format(dataset_name))
            self.class_names = read_classes(dataset_class_file)

        self.anchors = read_anchors(self.anchors_path)
        self.threshold = threshold# threshold
        self.ignore_thresh = ignore_thresh
        self.INPUT_SIZE = (Input_shape, Input_shape)  # fixed size or (None, None)
        self.is_fixed_size = self.INPUT_SIZE != (None, None)
        # LOADING SESSION...
        self.boxes, self.scores, self.classes, self.sess = self.load()


    def load(self):
        # Remove nodes from graph or reset entire default graph
        tf.reset_default_graph()

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x *1.0/ len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.x = tf.placeholder(tf.float32, shape=[None, Input_shape, Input_shape, channels])
        self.image_shape = tf.placeholder(tf.float32, shape=[2,])

        # Generate output tensor targets for filtered bounding boxes.
        scale1, scale2, scale3 = YOLOv3(self.x, len(self.class_names), trainable=self.trainable).feature_extractor()
        scale_total = [scale1, scale2, scale3]

        # detect
        boxes, scores, classes = predict(scale_total, self.anchors, len(self.class_names), self.image_shape, max_boxes=max_boxes,
                                         score_threshold=self.threshold, iou_threshold=self.ignore_thresh)

        # Add ops to save and restore all the variables
        saver = tf.train.Saver(var_list=None if self.COCO==True else tf.trainable_variables())

        # Allowing GPU memory growth
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # Initialize tf session
        sess = tf.Session(config = config)
        sess.run(tf.global_variables_initializer())
        
        # # For the case of COCO
        if(self.trainable):
            epoch = num_epochs if self.COCO == False else 2000
            checkpoint = model_checkpoint_path + '/model.ckpt-' + str(epoch-1)
            try:
                aaa = checkpoint + '.meta'
                my_abs_path = Path(aaa).resolve()
            except FileNotFoundError:
                print("Not yet training!")
            else:
                saver.restore(sess, checkpoint)
                print("checkpoint: ", checkpoint)
                print("already training!")
        
        return boxes, scores, classes, sess

    def detect_image(self, image):
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x *1.0/ len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.
        if self.is_fixed_size:
            assert self.INPUT_SIZE[0] % 32 == 0, 'Multiples of 32 required'
            assert self.INPUT_SIZE[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image, image_shape = letterbox_image(image, tuple(reversed(self.INPUT_SIZE)))
            # boxed_image, image_shape = resize_image(image, tuple(reversed(self.INPUT_SIZE)))
            #boxed_image.save("/home/yz2499/v2_yolo3/test.jpg")
        else:
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image, image_shape = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        inputs = np.expand_dims(image_data, 0)  # Add batch dimension.
        out_boxes, out_scores, out_classes = self.sess.run([self.boxes, self.scores, self.classes],
                                                           feed_dict={self.x: inputs,
                                                                      self.image_shape: image_shape,
                                                                      })
        print('Found {} boxes in the image'.format(len(out_boxes)))
        left = 0
        top = 0
        right = 0
        bottom = 0

        # Visualisation########################################################################################
        font = ImageFont.truetype(font=font_file, size=np.floor(3e-2 * image.size[1] + 0.5).astype(np.int32))
        thickness = (image.size[0] + image.size[1]) // 500  # do day cua BB

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box  # y_min, x_min, y_max, x_max
            top = max(0, np.floor(top + 0.5).astype(np.int32))
            left = max(0, np.floor(left + 0.5).astype(np.int32))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype(np.int32))
            right = min(image.size[0], np.floor(right + 0.5).astype(np.int32))
            print(label, (left, top), (right, bottom))  # (x_min, y_min), (x_max, y_max)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for j in range(thickness):
                draw.rectangle([left + j, top + j, right - j, bottom - j], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        return image, left, top, right, bottom


def parse_xml(path_to_file_name):
    import xml.etree.ElementTree as ET
    # parse the xml file into a tree
    tree = ET.parse(path_to_file_name)
    # get the root of the tree. In DAC case, the root is <annotation>
    root = tree.getroot()
    # the first level
    file_name = str(root.find("filename").text)
    size = root.find("size")
    obj = root.find("object")
    # the second level
    width = float(size.find("width").text)
    height = float(size.find("height").text)
    class_name = str(obj.find("name").text)
    bndbox = obj.find("bndbox")
    # the third level
    xmin = str(bndbox.find("xmin").text)
    xmax = str(bndbox.find("xmax").text)
    ymin = str(bndbox.find("ymin").text)
    ymax = str(bndbox.find("ymax").text)

    return file_name, class_name, xmin, ymin, xmax, ymax


def detect_img(yolo, batch_number, input_img='', ):
    output_name = input_img.split("/")[-1]
    try:
        image = Image.open(input_img)
    except:
        print('Open Error! Try again!')
    else:
        r_image, left, top, right, bottom = yolo.detect_image(image)
        # r_image.save(output_test_img_path+'/'+'result_'+output_name)
    if(batch_number<0):
        yolo.sess.close()
    return left, top, right, bottom

def detect_one_batch(batch_number, input_img=''):
    if(args.train_batch>=0):
        annotation_path = train_file_path + '/' + 'train_batch_{}'.format(batch_number) + '.txt'
    else:
        annotation_path = val_file_path + '/' + 'val_batch_{}'.format(batch_number) + '.txt'
    yolo_model = YOLO()
    iou = []
    miss = 0
    total_img = 0
    with open(annotation_path) as f:
        GG = f.readlines()
        # np.random.shuffle(GG)
        for line in (GG):
            line = line.split('$$')
            filename = line[0]
            if filename[-1] == '\n':
                filename = filename[:-1]
            left, top, right, bottom = detect_img(yolo_model, batch_number, filename)
            if(left==0 and top==0 and right==0 and bottom==0):
                miss += 1
            else:
                _, _, xmin, ymin, xmax, ymax = parse_xml(filename[:-3]+'xml')
                iou.append(calc_iou(left, top, right, bottom, int(xmin), int(ymin), int(xmax), int(ymax)))
            total_img += 1
        f.close()
    #print(iou)
    print("average iou is: {}".format(sum(iou)/len(iou)))
    print("miss rate is: {}".format(miss*1.0/total_img))

    def calc_iou(left,top,right,bottom,xmin,ymin,xmax,ymax):
        intersect_left = np.max(left,xmin)
        intersect_right = np.min(right,xmax)
        intersect_top = np.min(top,ymax)
        intersect_bottom = np.max(bottom,ymin)

        intersect_area = calc_area(intersect_left, intersect_right, intersect_top, intersect_bottom)
        b1_area = calc_area(left,right,top,bottom)
        b2_area = calc_area(xmin,xmax,ymax,ymin)

        iou = intersect_area/(b1_area + b2_area)
        return iou

    def calc_area(left,right,top,bottom):
        return (right - left)*(top - bottom)

if __name__ == '__main__':

    if args.image:
        if(args.train_batch>=0):
            print("Begin inference on train batch {}".format(args.train_batch))
            detect_one_batch(args.train_batch)
        elif(args.val_batch>=0):
            print("Begin inference on validation batch {}".format(args.val_batch))
            detect_one_batch(args.val_batch)
        else:
            detect_img(YOLO(), args.train_batch, input_test_img_path+'/'+args.name)
    else: 
        print("There's nothing I can do for you!\n")

