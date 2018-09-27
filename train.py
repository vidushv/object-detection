# from configuration.config import Input_shape, channels, path, batch_size, epochs
from configuration.config import Input_shape, channels, batch_size, num_epochs, visible_GPU, Image_size
from configuration.config import project_path, dataset_class_file, anchors_path, summary_writing_path
from configuration.config import train_file_path, val_file_path, model_checkpoint_path, ignore_thresh, threshold

from model.net import YOLOv3
from model.loss_function import compute_loss
from model.detect_function import predict
from utils.yolo_utils import get_training_data, read_anchors, read_classes, get_dac_batch_data

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import numpy as np
import tensorflow as tf
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = visible_GPU

np.random.seed(101)


# PATH = path + '/yolo3'
classes_data = read_classes(dataset_class_file)
anchors = read_anchors(anchors_path)

data_path_train = 'Do_not_use_for_now'
data_path_valid = 'Do_not_use_for_now'
data_path_test = 'Do_not_use_for_now'


input_shape = (Input_shape, Input_shape)  # multiple of 32
########################################################################################################################
"""
# Clear the current graph in each run, to avoid variable duplication
# tf.reset_default_graph()
"""
print("Starting 1st session...")
# Explicitly create a Graph object
graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Start running operations on the Graph.
    # STEP 1: Input data ###############################################################################################

    X = tf.placeholder(tf.float32, shape=[None, Input_shape, Input_shape, channels], name='Input')  # for image_data
    with tf.name_scope("Target"):
        Y1 = tf.placeholder(tf.float32, shape=[None, Input_shape/32, Input_shape/32, 3, (5+len(classes_data))], name='target_S1')
        Y2 = tf.placeholder(tf.float32, shape=[None, Input_shape/16, Input_shape/16, 3, (5+len(classes_data))], name='target_S2')
        Y3 = tf.placeholder(tf.float32, shape=[None, Input_shape/8, Input_shape/8, 3, (5+len(classes_data))], name='target_S3')
        # Y = tf.placeholder(tf.float32, shape=[None, 100, 5])  # for box_data
    # Reshape images for visualization
    x_reshape = tf.reshape(X, [-1, Input_shape, Input_shape, 1])
    tf.summary.image("input", x_reshape)
    # STEP 2: Building the graph #######################################################################################
    # Building the graph
    # Generate output tensor targets for filtered bounding boxes.
    scale1, scale2, scale3 = YOLOv3(X, len(classes_data)).feature_extractor()
    scale_total = [scale1, scale2, scale3]

    with tf.name_scope("Loss_and_Detect"):
        # Label
        y_predict = [Y1, Y2, Y3]
        # Calculate loss
        loss = compute_loss(scale_total, y_predict, anchors, len(classes_data), print_loss=False)


        image_shappe = tf.placeholder(tf.float32, shape=[2,])
        boxes, scores, classes = predict(scale_total, anchors, len(classes_data), image_shappe,
                                         score_threshold=threshold, iou_threshold=ignore_thresh)



        # loss_print = compute_loss(scale_total, y_predict, anchors, len(classes_data), print_loss=False)
        tf.summary.scalar("Loss", loss)
    with tf.name_scope("Optimizer"):
        # optimizer
        # for VOC: lr:0.0001, decay:0.0003 with RMSProOp after 60 epochs
        # learning_rate = tf.placeholder(tf.float32, shape=[1], name='lr')
        # decay = 0.0003
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss, global_step=global_step)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay).minimize(loss)
        # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.01).minimize(loss)
    # STEP 3: Build the evaluation step ####################################################################
    # with tf.name_scope("Accuracy"):
    #     # Model evaluation
    #     accuracy = 1  #
    # STEP 4: Merge all summaries for Tensorboard generation ###############################################
    # create a saver
    saver = tf.train.Saver()

    # STEP 5: Train the model, and write summaries #########################################################
    # The Graph to be launched (described above)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    with tf.Session(config=config, graph=graph) as sess:
        # Merges all summaries collected in the default graph
        summary_op = tf.summary.merge_all()
        # Summary Writers
        if(not os.path.isdir(summary_writing_path)):
            os.mkdir(summary_writing_path)
        train_summary_writer = tf.summary.FileWriter(summary_writing_path, sess.graph)
        validation_summary_writer = tf.summary.FileWriter(summary_writing_path, sess.graph)

        sess.run(tf.global_variables_initializer())

        # If you want to continue training from check point
        # checkpoint = "/home/minh/PycharmProjects/yolo3/save_model/SAVER_MODEL_boatM/model.ckpt-" + "1"
        # saver.restore(sess, checkpoint)

        # epochs = 50  #
        # batch_size = 32  # consider
        # num_batches = 2922
        num_train_batches = len(os.listdir(train_file_path))
        num_val_batches = len(os.listdir(val_file_path))
        best_loss_valid = 10e6
        for epoch in range(num_epochs):
            print("train on epoch {}".format(epoch))
            start_time = time.time()
            ## Training#######################################################################################
            mean_loss_train = []
            for batch in range(num_train_batches):
                x_train, box_data_train, image_shape_train, y_train = get_dac_batch_data(batch, data_path_train,
                    input_shape, anchors, train=True, num_classes=len(classes_data), max_boxes=20, load_previous=False)

                summary_train, loss_train, _ = sess.run([summary_op, loss, optimizer],
                                                        feed_dict={X: (x_train/255.),
                                                                   Y1: y_train[0],
                                                                   Y2: y_train[1],
                                                                   Y3: y_train[2]})  # , options=run_options)
                train_summary_writer.add_summary(summary_train, epoch)
                # Flushes the event file to disk
                train_summary_writer.flush()
                # summary_writer.add_summary(summary_train, counter)
                mean_loss_train.append(loss_train)
                # calculate time
                duration = time.time() - start_time
                start_time = time.time()
                print("(batch: {}, \tepoch: {})\tloss: {:.2f}\ttime: {:.1f}".format(batch+1, epoch + 1, loss_train, duration))

            # summary_writer.add_summary(summary_train, global_step=epoch)
            mean_loss_train = np.mean(mean_loss_train)

            # Validation #####################################################################################
            print("Begin validating...")
            start_time = time.time()
            mean_loss_valid = []
            for batch in range(num_val_batches):
                x_valid, box_data_valid, image_shape_valid, y_valid = get_dac_batch_data(batch, data_path_valid,
                    input_shape, anchors,train=False, num_classes=len(classes_data), max_boxes=20, load_previous=False)
                # Run summaries and measure accuracy on validation set
                summary_valid, loss_valid = sess.run([summary_op, loss],
                                                    feed_dict={X: (x_valid/255.),
                                                               Y1: y_valid[0],
                                                               Y2: y_valid[1],
                                                               Y3: y_valid[2]})  # ,options=run_options)

                out_boxes, _, _ = sess.run([boxes, scores, classes],
                                                                   feed_dict={X: (x_valid/255.),
                                                                              image_shappe: np.array(Image_size)
                                                                              # self.is_training: False
                                                                              })
                validation_summary_writer.add_summary(summary_valid, epoch)
                # Flushes the event file to disk
                validation_summary_writer.flush()
                mean_loss_valid.append(loss_valid)
                print("{} boxes found in validation batch {}".format(len(out_boxes), batch))

            # calculate time
            duration = time.time() - start_time
            sec_per_batch = duration * 1.0 / num_val_batches
            mean_loss_valid = np.mean(mean_loss_valid)
            print("sec per batch while validation: {:.1f}".format(sec_per_batch))
            print("epoch {} / {} \ttrain_loss: {:.2f},\tvalid_loss: {:.2f}".format(epoch+1, num_epochs, mean_loss_train, mean_loss_valid))

            if(epoch % 10 == 0 or epoch == num_epochs-1):
                if(not os.path.isdir(model_checkpoint_path)):
                    print("creating a folder for saving model checkpoints...")
                    try:
                        os.mkdir(model_checkpoint_path)
                    except OSError:
                        print("Fail to create a folder for saving checkpoints!!")
                checkpoint_path = model_checkpoint_path + "/model.ckpt"
                saver.save(sess, checkpoint_path, global_step=epoch)
                print("Model saved in file: %s" % checkpoint_path)

        print("Tuning completed!")

        # Testing ######################################################################################################
        # mean_loss_test = []
        # for start in (range(0, 128, batch_size)):
        #     end = start + batch_size
        #     if end > number_image_train:
        #         end = number_image_train
        #     # Loss in test data set
        #     summary_test, loss_test = sess.run([summary_op, loss],
        #                                        feed_dict={X: (x_test[start:end]/255.),
        #                                                   Y1: y_test[0][start:end],
        #                                                   Y2: y_test[1][start:end],
        #                                                   Y3: y_test[2][start:end]})
        #     mean_loss_test.append(mean_loss_test)
        #     # print("Loss on test set: ", (loss_test))
        # mean_loss_test = np.mean(mean_loss_test)
        # print("Mean loss in all of test set: ", mean_loss_test)
        # summary_writer.flush()
        train_summary_writer.close()
        validation_summary_writer.close()
        # summary_writer.close()



