import numpy as np
import os, sys
import random
import argparse
import os.path, shutil
import xml.etree.ElementTree as ET

from configuration import config

'''
what we use in config.py:
-> batch_size
-> dataset_path
-> dataset_class_file
-> train_file_path
-> val_file_path
'''

# Add argument
# def argument():
#     parser = argparse.ArgumentParser(description='Choose the whole dataset or a subset.')
#     parser.add_argument('--extract', action='store_true', help='extract a subset of the dataset')
#     parser.add_argument('--train', type=int, default=0, help='number of train batches that will be extracted')
#     parser.add_argument('--val', type=int, default=0, help='number of validation batches that will be extracted')
#     args = parser.parse_args()
#     return args
# args = argument()
parser = argparse.ArgumentParser(description='Choose the whole dataset or a subset.')
parser.add_argument('--extract', action='store_true', help='extract a subset of the dataset')
parser.add_argument('--train', type=int, default=0, help='number of train batches that will be extracted')
parser.add_argument('--val', type=int, default=0, help='number of validation batches that will be extracted')
args = parser.parse_args()



'''
func    : classes()
input   : None
return  : content -> a list that contains all classes in the dataset
'''
def classes():
    class_file = open(config.dataset_class_file, 'r')
    content = class_file.readlines()
    class_file.close()
    content = [x.strip() for x in content]
    if (len(content[-1]) == 0 ):
        content.pop()
    return content
    
    
'''
func    : index_class_name()
input   : class_name  -> a class that we want to find its index
          classes     -> a list contains all class names
return  : index       -> the index of the class_name in dac_classes
'''
def index_class_name(class_name, all_classes):
    index = all_classes.index(class_name)
    return index

    
'''
func    : parse_xml
input   : path_to_file_name -> the absolute path to a xml file
return  : file_name         -> the name of the corresponding jpg file
          class_name        -> the class that the object belongs to
          xmin              -> xmin of the bounding box
          xmax              -> xmax of the bounding box
          ymin              -> ymin of the bounding box
          ymax              -> ymax of the bounding box
''' 
##########################################################
### Feel free to customize your own parse_xml function ###
##########################################################
def parse_xml(path_to_file_name):
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


def main():

    print('\n------------------')
    print('Begin generating data.txt ....\n')

    if(os.path.isfile(config.target_data_file)):
        print("data.txt file already exist")
    else:
        # extract classes of the dataset from txt file
        all_classes = classes()
        # flush/create the file
        open(config.target_data_file, 'w+').close()
        # traverse the dataset topdown
        for (root, dirs, files) in os.walk(config.dataset_path, topdown=True):
            print('Now going through the folder '+ root)
            for file in files:
                if '.xml' in file:
                    # extract elements that'll be written into file
                    xml_file_path = root + '/' + file
                    image_path = xml_file_path.replace('.xml', '.jpg')
                    file_name, class_name, xmin, ymin, xmax, ymax = parse_xml(xml_file_path)
                    index_class = index_class_name(class_name, all_classes)
                    # write into file
                    txt_file = open(config.target_data_file, 'a')
                    txt_file.write(image_path + '$$' + xmin + ',' + ymin + ',' + xmax + ',' + ymax + ',' + str(index_class) + '\n')
                    txt_file.close()


    # Separate the whole data.txt into train.txt and val.txt
    print('\nBegin separating the file into <train> and <val> ....')

    if(not os.path.isfile(config.target_data_file)):
        print("<dataset_name>_data.txt file disappears!")
    else:
        with open(config.target_data_file) as bigfile:
            random.seed(923)
            content = bigfile.readlines()
            # random shuffle the data
            np.random.shuffle(content)
            # get overall line number in data.txt
            num_lines = len(content)
            print("There are {} images in the dataset!".format(num_lines))
            num_batches = (num_lines//config.batch_size + 1) if num_lines % config.batch_size > 0 else num_lines//config.batch_size
            print("The dataset can be partitioned into {} batches in total.".format(num_batches))

            if(args.extract):
                print("\nExtract command detected. Now begin extracting a subset...")
                num_train_batches = args.train
                num_val_batches = args.val
                num_val = config.batch_size * num_val_batches
                num_train = config.batch_size * num_train_batches
                # generate line numbers in data.txt that represents for the validation set
                val_line_numbers = random.sample(range(0,num_val+num_train), num_val)
            else:
                # put 1/split_proportion of the whole data into validation set
                num_val = num_lines//config.split_proportion
                num_train = num_lines - num_val
                # compute number of train and val batches
                num_train_batches = (num_train//config.batch_size + 1) if num_train % config.batch_size > 0 else num_train//config.batch_size
                num_val_batches = (num_val//config.batch_size + 1) if num_val % config.batch_size > 0 else num_val//config.batch_size
                # generate line numbers in data.txt that represents for the validation set
                val_line_numbers = random.sample(range(0,num_lines), num_val)

            print("number of validation data = ",num_val)
            print("number of training data =", num_train)
            print("\nPrepare txt files...\n")
            # check if directories exist
            if(not os.path.isdir(config.train_file_path)):
                os.mkdir(config.train_file_path)
            if(not os.path.isdir(config.val_file_path)):
                os.mkdir(config.val_file_path)    
            # flush/create train.txt and val.txt        
            for i in range(num_train_batches):
                train_batch_file = config.train_file_path + '/train_batch_{}'.format(i) + '.txt'
                open(train_batch_file, 'w+').close()
            for i in range(num_val_batches):
                val_batch_file = config.val_file_path + '/val_batch_{}'.format(i) + '.txt'
                open(val_batch_file, 'w+').close()
            
            print("Begin appending data into separate txt files...\n")
            val_cnt = 0
            train_cnt = 0
            for linenum in range(num_lines):
                if(linenum % 1000 == 0):
                    print("Already read {} lines.".format(linenum))
                if(val_cnt>=num_val and train_cnt>=num_train):
                    break
                if linenum in val_line_numbers:
                    smallfile = open(config.val_file_path + '/val_batch_{}'.format(val_cnt%num_val_batches) + '.txt', "a")
                    val_cnt += 1
                else:
                    smallfile = open(config.train_file_path + '/train_batch_{}'.format(train_cnt%num_train_batches) + '.txt', "a")
                    train_cnt += 1
                smallfile.write(content[linenum])
                smallfile.close()
            print("{} data processed in total.".format(val_cnt+train_cnt))

        print('\nDone!\n')


if __name__ == '__main__':
    main()