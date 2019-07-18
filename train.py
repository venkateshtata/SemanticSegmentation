import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from PIL import Image

masks = glob.glob("./dataset/isbi2015/train/label/*.png")
orgs = glob.glob("./dataset/isbi2015/train/image/*.png")


imgs_list = []
masks_list = []
for image, mask in zip(orgs, masks):
    imgs_list.append(np.array(Image.open(image).resize((512,512))))
    
    im = Image.open(mask).resize((512,512))
    masks_list.append(np.array(im))

imgs_np = np.asarray(imgs_list)
masks_np = np.asarray(masks_list)

'''
print(imgs_np.shape, masks_np.shape)
'''

'''
from utils import plot_imgs
plot_imgs(org_imgs=imgs_np, mask_imgs=masks_np, nm_img_to_plot=10, figsize=6)
'''


#Get data into correct shape, dtype and range (0.0-1.0)

'''
print(imgs_np.max(), masks_np.max())
'''

x = np.asarray(imgs_np, dtype=np.float32)/255
y = np.asarray(masks_np, dtype=np.float32)/255


y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
'''
print(x.shape, y.shape)
'''
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
'''
print(x.shape, y.shape)
'''

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.5, random_state=0)
'''
print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_val: ", x_val.shape)
print("y_val: ", y_val.shape)
'''

#Prepare train generator with data augmentation

from utils import get_augmented

train_gen = get_augmented(
    x_train, y_train, batch_size=2,
    data_gen_args = dict(
        rotation_range=15.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=50,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant'
    ))

sample_batch = next(train_gen)
xx, yy = sample_batch
'''
print(xx.shape, yy.shape)
'''

'''
from keras_unet.utils import plot_imgs
plot_imgs(org_imgs=xx, mask_imgs=yy, nm_img_to_plot=2, figsize=6)
'''

#Initialize network

from unet_model import unet_model

input_shape = x_train[0].shape

model = unet_model(
    input_shape,
    num_classes=1,
    filters=64,
    dropout=0.2,
    num_layers=4,
    output_activation='sigmoid'
)


print(model.summary())




#Compile + train

from keras.callbacks import ModelCheckpoint


model_filename = 'segm_model_v0.h5'
callback_checkpoint = ModelCheckpoint(
    model_filename, 
    verbose=1, 
    monitor='val_loss', 
    save_best_only=True,
)

from keras.optimizers import Adam, SGD
from metrics import iou, iou_thresholded

model.compile(
    optimizer=SGD(lr=0.01, momentum=0.99),
    loss='binary_crossentropy',
    metrics=[iou, iou_thresholded]
)

history = model.fit_generator(
    train_gen,
    steps_per_epoch=100,
    epochs=10,
    
    validation_data=(x_val, y_val),
    callbacks=[callback_checkpoint]
)

