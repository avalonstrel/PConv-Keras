import gc
import datetime

import pandas as pd
import numpy as np

from copy import deepcopy
from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras import backend as K

import cv2
from PIL import Image
from libs.pconv_model import PConvUnet
from libs.util import random_mask


TRAIN_DIR = r"/unsullied/sharefs/linhangyu/Inpainting/Data/PlacesData/data_256"
TEST_DIR = "/unsullied/sharefs/linhangyu/Inpainting/Data/PlacesData/test_256"
VAL_DIR = "/unsullied/sharefs/linhangyu/Inpainting/Data/PlacesData/val_256"

BATCH_SIZE = 64

class DataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
        while True:

            # Get augmentend image samples
            ori = next(generator)

            # Get masks for each image sample
            mask = np.stack([random_mask(ori.shape[1], ori.shape[2]) for _ in range(ori.shape[0])], axis=0)

            # Apply masks to all image sample
            masked = deepcopy(ori)
            masked[mask==0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori


# Create training generator
train_datagen = DataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(512, 512), batch_size=BATCH_SIZE
)

# Create validation generator
val_datagen = DataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(512, 512), batch_size=BATCH_SIZE, seed=1
)

# Create testing generator
test_datagen = DataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(512, 512), batch_size=BATCH_SIZE, seed=1
)

test_data = next(test_generator)
(masked, mask), ori = test_data


def plot_callback(model):
    """Called at the end of each epoch, displaying our previous test images,
    as well as their masked predictions and saving them to disk"""

    # Get samples & Display them
    pred_img = model.predict([masked, mask])
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Clear current output and display test images
    combine_imgs = np.concatenate([pred_img, masked, mask, ori], axis=2)
    for i in range(len(combine_imgs)):
        img_array = combine_imgs[i]
        img = Image.fromarray(img_array.astype(np.uint8))
        img.save(r'data/train_samples/img_{}_{}.png'.format(i, pred_time))

print("Phase one training...")
# phase one
model = PConvUnet(weight_filepath='./model/logs/')
#model.load("./model/log/")
model.fit(
    train_generator,
    steps_per_epoch=10000,
    validation_data=val_generator,
    validation_steps=100,
    epochs=50,
    plot_callback=plot_callback,
    callbacks=[
        TensorBoard(log_dir='../data/logs/initial_training', write_graph=False)
    ]
)
print("Phase two training...")
#phase two without bn
model = PConvUnet(weight_filepath='model/logs/')
model.load(
    "./model/logs/latest_weights.h5",
    train_bn=False,
    lr=0.00005
)
# Run training for certain amount of epochs
model.fit(
    train_generator,
    steps_per_epoch=10000,
    validation_data=val_generator,
    validation_steps=100,
    epochs=20,
    workers=3,
    plot_callback=plot_callback,
    callbacks=[
        TensorBoard(log_dir='../data/logs/fine_tuning', write_graph=False)
    ]
)

# Test
#Load weights from previous run
model = PConvUnet(weight_filepath='model/logs/')
model.load(
    "./model/logs/latest_weights.h5",
    train_bn=False,
    lr=0.00005
)


n = 0
for (masked, mask), ori in tqdm(test_generator):

    # Run predictions for this batch of images
    pred_img = model.predict([masked, mask])
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    combine_imgs = np.concatenate([pred_img, masked, mask, ori], axis=2)
    # Clear current output and display test images
    for i in range(len(ori)):
        img_array = combine_imgs[i]
        img = Image.fromarray(img_array.astype(np.uint8))
        img.save(r'data/test_samples/img_{}_{}.png'.format(n, pred_time))
        n += 1

    # Only create predictions for about 100 images
    if n > 100:
        break
