# from argparse import ArgumentParser

from configuration.config import Input_shape, channels, threshold, ignore_thresh, visible_GPU, num_epochs
from configuration.config import anchors_path, dataset_name, dataset_class_file, model_checkpoint_path, font_file
from configuration.config import input_test_img_path, output_test_img_path, train_file_path, val_file_path
from model.net import YOLOv3
from model.detect_function import predict
from utils.yolo_utils import read_anchors, read_classes, letterbox_image  # , resize_image
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

    # @staticmethod
    # def argument():
    #     parser = argparse.ArgumentParser(description='COCO or VOC or customized dataset')
    #     parser.add_argument('--COCO', action='store_true', help='COCO flag')
    #     # parser.add_argument('--other', action='store_true', help='customized dataset flag')
    #     parser.add_argument('--image', action='store_true', help='image detection')
    #     parser.add_argument('--name', type=str, default='', help='name of the test image')
    #     args = parser.parse_args()
    #     return args

    def load(self):
        # Remove nodes from graph or reset entire default graph
        tf.reset_default_graph()

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.x = tf.placeholder(tf.float32, shape=[None, Input_shape, Input_shape, channels])
        self.image_shape = tf.placeholder(tf.float32, shape=[2,])
        # self.is_training = tf.placeholder(tf.bool)
        # image_shape = np.array([image.size[0], image.size[1]])  # tf.placeholder(tf.float32, shape=[2,])

        # Generate output tensor targets for filtered bounding boxes.
        # scale1, scale2, scale3 = YOLOv3(self.x, len(self.class_names), trainable=self.trainable, is_training=self.is_training).feature_extractor()
        scale1, scale2, scale3 = YOLOv3(self.x, len(self.class_names), trainable=self.trainable).feature_extractor()
        scale_total = [scale1, scale2, scale3]

        # detect
        boxes, scores, classes = predict(scale_total, self.anchors, len(self.class_names), self.image_shape,
                                         score_threshold=self.threshold, iou_threshold=self.ignore_thresh)

        # Add ops to save and restore all the variables
        saver = tf.train.Saver(var_list=None if self.COCO==True else tf.trainable_variables())

        # Allowing GPU memory growth
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # Initialize tf session
        sess = tf.Session(config = config)
        sess.run(tf.global_variables_initializer())
        
        # # epoch = input('Entrer a check point at epoch:')
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
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
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
        else:
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image, image_shape = letterbox_image(image, new_image_size)
            # boxed_image, image_shape = resize_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print("heights, widths:", image_shape)
        image_data /= 255.
        inputs = np.expand_dims(image_data, 0)  # Add batch dimension. #

        out_boxes, out_scores, out_classes = self.sess.run([self.boxes, self.scores, self.classes],
                                                           feed_dict={self.x: inputs,
                                                                      self.image_shape: image_shape,
                                                                      # self.is_training: False
                                                                      })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        left = 0
        right = 0
        top = 0
        bottom = 0

        # Visualisation#################################################################################################
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
    # get the root of the tree. In dac case, the root is <annotation>
    root = tree.getroot()
    # get the first level
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

def detect_video(yolo, video_path=None, output_video=None):
    import urllib.request as urllib
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 20.0  # display fps frame per second
    accum_time = 0
    curr_fps = 0
    prev_time = timer()
    if video_path=='stream':
        url = 'http://10.18.97.1:8080/shot.jpg'
        out = cv2.VideoWriter(output_video, fourcc, fps, (1280, 720))
        while True:

            # Use urllib to get the image and convert into a cv2 usable format
            imgResp = urllib.urlopen(url)
            imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgNp, -1)
            # print(np.shape(img))  # get w, h from here

            image = Image.fromarray(img)
            image = yolo.detect_image(image)
            result = np.asarray(image)

            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("Result", result)
            out.write(result)

            # To give the processor some less stress
            # time.sleep(0.1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        out.release()
        # Closes all the frames
        cv2.destroyAllWindows()

        yolo.sess.close()
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")
        # The size of the frames to write
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
        while True:
            ret, frame = cap.read()
            if ret==True:
                image = Image.fromarray(frame)

                image = yolo.detect_image(image)
                result = np.asarray(image)

                curr_time = timer()
                exec_time = curr_time - prev_time
                prev_time = curr_time
                accum_time = accum_time + exec_time
                curr_fps = curr_fps + 1
                if accum_time > 1:
                    accum_time = accum_time - 1
                    fps = "FPS: " + str(curr_fps)
                    curr_fps = 0
                cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.50, color=(255, 0, 0), thickness=2)
                cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
                cv2.imshow("Result", result)

                out.write(result)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        out.release()
        # Closes all the frames
        cv2.destroyAllWindows()

        yolo.sess.close()

def detect_img(yolo, batch_number, input_img='', ):
    output_name = input_img.split("/")[-1]
    try:
        image = Image.open(input_img)
    except:
	    print('Open Error! Try again!')
    else:
        r_image, left, top, right, bottom = yolo.detect_image(image)
        r_image.save(output_test_img_path+'/'+'result_'+output_name)
        # r_image.show()
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
            if (left==0 and right==0 and top==0 and bottom==0 ):
                miss += 1
            else:
                _, _, xmin, ymin, xmax, ymax = parse_xml(filename[:-3]+'xml')
                cur_iou = calc_iou(left, top, right, bottom, int(xmin), int(ymin), int(xmax), int(ymax))
                iou.append(cur_iou)
                print("IOU:{}".format(cur_iou))
            total_img += 1
        f.close()
        print("average IOU is: {}".format(sum(iou)/len(iou)))
        print("miss rate is: {}".format(miss*1.0/total_img))

def calc_iou(left,top,right,bottom,xmin,ymin,xmax,ymax):
    #print("left:{}, top:{}, right:{}, bottom:{}".format(left, top, right, bottom))
    #print("xmin:{}, ymin:{}, xmax:{}, ymax:{}".format(xmin, ymin, xmax, ymax))
    intersect_left = max(left,xmin)
    intersect_right = min(right,xmax)
    intersect_top = max(top,ymin)
    intersect_bottom = min(bottom,ymax)

    intersect_area = calc_area(intersect_left, intersect_right, intersect_top, intersect_bottom)
    b1_area = calc_area(left,right,top,bottom)
    b2_area = calc_area(xmin,xmax,ymin,ymax)

    iou = intersect_area/(b1_area + b2_area - intersect_area)
    return iou

def calc_area(left,right,top,bottom):
    return (right - left)*(bottom - top)

if __name__ == '__main__':
    # yolov3 = YOLO()
    # if yolov3.args.image:
    #     input_image = input_test_img_path + '/' + yolov3.args.name
    #     output = output_test_img_path + '/' + 'result_' + yolov3.args.name
    #     detect_img(yolov3, input_img=input_image, output_img=output)
    # else:
    #     video_path = sys.argv[3]
    #     output = sys.argv[4]
    #     detect_video(YOLO(), video_path, output)

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

