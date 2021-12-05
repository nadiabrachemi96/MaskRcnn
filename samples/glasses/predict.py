import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
 
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
 
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
 
import glasses
 
#%matplotlib inline 
 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
 
# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
BALLON_WEIGHTS_PATH = "../../mask_rcnn_glasses_0002.h5"  # TODO: update this path
 
config = glasses.glassesConfig()
glasses_DIR = os.path.join(ROOT_DIR, "dataset")
 
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
config = InferenceConfig()
config.display()
 
# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
 
# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"
 
def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
     
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
 
# Load validation dataset
dataset = glasses.glassesDataset()
dataset.load_glasses(glasses_DIR, "val")
 
# Must call before using the dataset
dataset.prepare()
 
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
 
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
 
weights_path = model.find_last()
 
# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

#########################################################################################################
image1 = mpimg.imread('C:/Users/LENOVO/mask/my_project_v2/dataset/val/T10.JPG')
    # Run object detection
print(len([image1]))
results1 = model.detect([image1], verbose=1)
 
    # Display results
ax = get_ax(1)
r1 = results1[0]
visualize.display_instances(image1, r1['rois'], r1['masks'], r1['class_ids'],dataset.class_names, r1['scores'], ax=ax,title="Predictions1")