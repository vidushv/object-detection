from network_function import *
from detect_function import *
from config import anchors
# TEST BUILD NETWORK
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test():
    box1 = tf.random_normal([1, 13, 13, 3, 2], mean=1, stddev=4, seed=1)
    box2 = tf.random_normal([1, 13, 13, 3, 2], mean=1, stddev=4, seed=1)

    mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = np.array([416, 416])
    image_shape = np.array([1028, 516])
    a = yolo_correct_boxes(box1, box2, input_shape, image_shape)

    # x = tf.shape(a)
    writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
    # python network_function.py
    # tensorboard --logdir="./graphs" --port 6006
    # these log file is saved in graphs folder, can delete these older log file
    # porte 6006 may be change
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(tf.shape(box1)))
        print(sess.run(tf.shape(box2)))
        print(sess.run(tf.shape(a)))
        # print(sess.run(a[0][0][0][0][:]))
    writer.close()
    sess.close()
    return 0


test()