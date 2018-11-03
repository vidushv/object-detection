# Object Detection - YOLOv3 - Tensorflow

This project is about object detection based on YOLOv3 network. The original YOLOv3 tendorflow code is forked from [here](https://github.com/maiminh1996/YOLOv3-tensorflow). I rearranged the folder system and made some changes in the code to make it cleaner and easier for training and testing on a customized dataset.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. Any bugs, contact me as soon as possible! Thanks!

### Prerequisites

```
python - 2.7.14
numpy - 1.13.3
h5py - 2.8.0
tensorflow - 1.10.1
keras - 2.2.2
pillow - 5.2.0
matplotlib - 2.1.1
```

If some packages listed above are not in your list, just type in
```
pip install package_name
```
If you are on a server(like me), cd to /home/userid, and then type in
```
pip install package_name --user
```

## Running the tests

I'll explain the way of running my code here.

### Download pre-trained weights by author of YOLO network.

Download the pre-trained weights from my google drive and put it anywhere you want. I suggest you put it into ./model directory.

```
cd ~
git clone https://github.com/chentinghao/download_google_drive.git
cd download_google_drive
python download_gdrive.py 1cVWJE1hv1M_KxzyJN6NE52L2JKqjW133 /path/to/project/model/directory
```

The script I used to download files from google drive is [here](https://github.com/chentinghao/download_google_drive).

### Configuration

The configuration of the model is stored in ./configuration/config.py .
Every time when you do inference or train on your own dataset, make sure that you have set correct paths or dataset name in the file config.py. This configuration file is well commented, you can customize everything that you want.

### Data_manager

'data_manager.py' in the project folder is used to generate training and validation data for your customized dataset. Morever, it can also help you extract a small subset from a larger one to help you train easier. The code will first traverse your dataset folder, create a txt file that contains all of your absolute image paths and bounding box information, and then partition them into multiple batches storing in './data/dataset_train' or './data/dataset_val' correspondingly. To use the data manager, follow these steps:

Step 1: Include a classes.txt in your dataset folder. Change the paths to your dataset in './configuration/config.py'.

Step 2: Run data_manager.py

If you want to generate the training and validationd data for the whole dataset, type in
```
python data_manager.py
```

If you want to extract a subset, say, 90 training batches and 10 validation batches from the whole dataset:
```
python data_manager.py --extract --train 90 --val 10
```

I'll include some templetes in './data/dataset_templete/' to tell you what your dataset should look like.

### Inference

Run the following steps, you'll be able to do inference on images.

Step 1: Clean the image folder

```
bash clean_result
```

Step 2: Copy some images into the ./test_img/input folder.

For example:
```
cp /path/to/your/image.jpg /home/yichi/v2_yolo3/test_img/input
```

Step 3: Run propagation.py

There are five arguments that you can pass to propagetion.py.

--COCO: Use the pre-trained weights by the author of Yolov3 on COCO dataset.
		No this argument means you'll use your own trained weights stored in ./save_model/checkpoint/

--image: Tell the code you want to detect on images.

--name: Name of the image that you want to run the code on.

--train_batch: Do inference on a whole trianing batch.

--val_batch: Do inference on a whole validation batch.

For example, you want to run the detection model on an image named as 'person.jpg' in './test_img/input' folder with your own training weights. The type in
```
python propagation.py --image --name person.jpg
```

If you want to run the detection model on validation batch 1 that you have extracted in the './data/' folder with pre-trained weights from the Yolov3 author, type in
```
python propagation.py --image --val_batch 1
```

You can view the result image in the './test_img/output' folder.

### Train your own model

Make sure you have set the correct paths in configuration file, and then run
```
python train.py
```

### Clean your project

If you want to flush away all input and output test images, as well as these generated training and validation data, those model checkpoints, type in
```
bash clean
bash clean_result
```

It will give you a 'clean' project. You'll be able to train on different datasets again!

## Results

<img src="/test_img/positive_results/0.jpg" />
<img src="/test_img/positive_results/1.jpg" />

## Authors

* **Yichi Zhang** -  [Web](http://zhang.ece.cornell.edu/people/yichi-zhang/)
