# CONFIGURE
visible_GPU = "0"

##########################################################
# change the path according to your project settings
dataset_name = 'dac'
project_path = '/home/vv259/object-detection'
dataset_path = '/home/vv259/dataset/data_training_March'
yolov3_h5_file = '/home/vv259/object-detection/model/yolov3.h5'
##########################################################



# yolov3 anchor file path
anchors_path = project_path+'/model/' + dataset_name + '_anchors.txt'
# the txt file stores data
target_data_file = project_path+'/data/'+dataset_name+'_data.txt'
# the txt file that stores classes in the dataset
dataset_class_file = project_path+ '/data/' + dataset_name + '_classes.txt'
# the place that save splitted train file
train_file_path = project_path+'/data/'+dataset_name+'_train'
# the place that saves splitted val file
val_file_path = project_path+'/data/'+dataset_name+'_val'
# the place that writes summary
summary_writing_path = project_path+'/save_model/graphics'
# the place that saves model checkpoints
model_checkpoint_path = project_path+'/save_model/'+dataset_name+'_CHECKPOINTS'
# font file for visualization
font_file = project_path+'/font/FiraMono-Medium.otf'
# the place that stores input test images
input_test_img_path = project_path+'/test_img/input'
# the place that stores results of input test images
output_test_img_path = project_path+'/test_img/output'


Image_size = [360,640]

# split the whole dataset into training data and validation data
# the proporiton is   train : val = split_proportion-1 : 1
split_proportion = 7

# image pre-processing
Input_shape = 416  # width=height # 608 or 416 or 320
channels = 3  # RBG
angle = 0
saturation = 1.5
exposure = 1.5
hue = 0.1
jitter = 0.3
random = 1


num_epochs = 50
batch_size = 16
threshold = 0.3
ignore_thresh = 0.5
truth_thresh = 1
momentum = 0.9
decay = 0.0005
learning_rate = 0.001
burn_in = 1000
max_batches = 500200

# policy=steps
learning_rate_steps = [40000, 45000]  # steps=400000,450000
learning_rate_scales = [0.1, 0.1]  # scales=.1,.1
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
