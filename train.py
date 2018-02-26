import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import keras.backend as K
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth=True
sess = tf.Session(config=tfconfig)
K.set_session(sess)


import numpy as np
import glob
import random
from PIL import Image
import sys
import generate_data_lap as gd_lap
import generate_data_emo_gender as gd_emo

import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.generic_utils import Progbar
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, CSVLogger
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, Dense, Flatten
from keras.layers import Reshape, TimeDistributed, Activation
from keras.layers.pooling import GlobalAveragePooling2D


from tqdm import tqdm
import pickle

w0,h0=256, 256
crop_size=224
batch_size = 32
batch_size_task1 = batch_size
batch_size_task2 = batch_size

#base_model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(crop_size, crop_size, 3))
base_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(crop_size, crop_size, 3))

common_feat = Flatten()(base_model.get_output_at(0))

image_batch_t1 = Input(shape=(crop_size, crop_size, 3), name='in_t1')
image_batch_t2 = Input(shape=(crop_size, crop_size, 3), name='in_t2')

common_feat_t1 = GlobalAveragePooling2D()(base_model(image_batch_t1))
common_feat_t2 = GlobalAveragePooling2D()(base_model(image_batch_t2))


num_outputs_age=101
num_outputs_gender=2
num_outputs_smile=2
num_outputs_glass=2


pred_age = Dense(name='age', units=num_outputs_age, kernel_initializer="he_normal", activation="softmax")(common_feat_t1)
pred_gender = Dense(name='gender', units=num_outputs_gender, kernel_initializer="he_normal", activation="softmax")(common_feat_t2)
pred_smile = Dense(name='smile', units=num_outputs_smile, kernel_initializer="he_normal", activation="softmax")(common_feat_t2)
pred_glass = Dense(name='glass', units=num_outputs_glass, kernel_initializer="he_normal", activation="softmax")(common_feat_t2)


model=Model(input = [image_batch_t1, image_batch_t2], 
            output = [pred_age, pred_gender, pred_smile, pred_glass])

base_lr = 0.0001
rmsprop=keras.optimizers.Adam(lr=base_lr)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

list_dat_train_age = gd_lap.get_train_list()
list_dat_valid_age = gd_lap.get_val_list(cast_age_to_int=True)

list_dat_train_emo = gd_emo.get_train_list()
list_dat_valid_emo = gd_emo.get_val_list()

gen_train_age = gd_lap.generate_data(list_dat_train_age, batch_size_task1, h0, w0, crop_size, SHUFFLE=True, RANDOM_CROP=True)
gen_valid_age = gd_lap.generate_data(list_dat_valid_age, batch_size_task1, h0, w0, crop_size, SHUFFLE=True, RANDOM_CROP=False)

gen_train_emo = gd_emo.generate_data(list_dat_train_emo, batch_size_task2, h0, w0, crop_size, SHUFFLE=True, RANDOM_CROP=True)
gen_valid_emo = gd_emo.generate_data(list_dat_valid_emo, batch_size_task2, h0, w0, crop_size, SHUFFLE=True, RANDOM_CROP=False)


EPOCH_NB = 200
BATCH_NB = np.max((len(list_dat_train_age)/batch_size_task1, len(list_dat_train_emo)/batch_size_task2))
BATCH_NB_TEST = np.min((len(list_dat_valid_age)/batch_size_task1, len(list_dat_valid_emo)/batch_size_task2))
lr_schedule = [base_lr]*100 + [base_lr*0.1]*50 + [base_lr*0.01]*50


LOSSES = []
LOSSES_test = []
LOSSES.append(model.metrics_names)
LOSSES_test.append(model.metrics_names)
for epoch_nb in xrange(EPOCH_NB):
    if epoch_nb>1 and lr_schedule[epoch_nb] != lr_schedule[epoch_nb-1]:
        model.save('model_epoch_%d_weights.h5'%epoch_nb)
        K.get_session().run(model.optimizer.lr.assign(lr_schedule[epoch_nb]))
    Losses=[]
    for batch_nb in tqdm(xrange(BATCH_NB), total=BATCH_NB):
        [Image_data_age, Labels_age] = gen_train_age.next()
        [Image_data_emo, Labels_emo] = gen_train_emo.next()
        losses = model.train_on_batch([Image_data_age, Image_data_emo],
                                      [Labels_age['age'], Labels_emo['gender'],
                                       Labels_emo['smile'], Labels_emo['glass']])
        Losses.append(losses)
        if epoch_nb == 0: print losses
    Losses_test=[]
    for batch_nb in tqdm(xrange(BATCH_NB_TEST)):
        [Image_data_age, Labels_age] = gen_valid_age.next()
        [Image_data_emo, Labels_emo] = gen_valid_emo.next()
        losses = model.test_on_batch([Image_data_age, Image_data_emo],
                                      [Labels_age['age'], Labels_emo['gender'],
                                       Labels_emo['smile'], Labels_emo['glass']])
        Losses_test.append(losses)


    print 'Done for Epoch %d.'% epoch_nb
    print model.metrics_names
    print np.array(Losses).mean(axis=0)
    print np.array(Losses_test).mean(axis=0)
    LOSSES.append(Losses)
    LOSSES_test.append(Losses_test)
    pickle.dump({'LOSSES':LOSSES, 'LOSSES_test':LOSSES_test} , open("historty.pickle", 'w'))

model.save('model_last_weights.h5')

    
