from PIL import Image
import numpy as np
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

data_gen_args = dict(featurewise_center=False,
                            samplewise_center=False,
                            featurewise_std_normalization=False,
                            samplewise_std_normalization=False,
                            zca_whitening=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            shear_range=0.08,  # 5 degree
                            zoom_range=0.1,
                            channel_shift_range=0.,
                            fill_mode='bilinear',
                            cval=0.,
                            horizontal_flip=True,
                            vertical_flip=False,
                            rescale=None)


root_dir = '/path/to/LAP_data/'

train_src = root_dir + 'data/LAP/age_trainval_list_det_crop.txt'
val_src   = root_dir + 'data/LAP/age_test_list_det_crop.txt'
img_src   = root_dir + 'data/LAP/images/'

def get_list(src, cast_age_to_int=True):
    list_dat=[]
    for line in open(src).readlines():
        items=line.split()
        rect=[float(f) for f in items[3:7]]
        rect[2]=rect[0]+rect[2]-1
        rect[3]=rect[1]+rect[3]-1
        if cast_age_to_int:
            age = int(np.round(float(items[1])))
        else:
            age = float(items[1])
        age_std = float(items[2]) 
        if age_std == 0:
            age_std = 1e-10
        if age>100:
            continue
        list_dat.append({'add': items[0], 
                         'age': age, 
                         'age_std': age_std,
                         'rect': rect} )
    return list_dat


def get_train_list(cast_age_to_int=True):
    return get_list(train_src, cast_age_to_int)

def get_val_list(cast_age_to_int=False):
    return get_list(val_src, cast_age_to_int)


def generate_data(list_dat, batch_size, h0, w0, crop_size, SHUFFLE=True, RANDOM_CROP=True, TRAINING_PHASE=True):
    #image_gen=ImageDataGenerator(**data_gen_args)
    batch_nb=-1
    batch_total=len(list_dat)/batch_size

    assert(h0>=crop_size)
    assert(w0>=crop_size)
    h_max=h0-crop_size
    w_max=w0-crop_size
    h_max_half=int(0.5*h_max)
    w_max_half=int(0.5*w_max)
    if SHUFFLE: random.shuffle(list_dat)
    while 1:
        batch_nb+=1
        if batch_nb==batch_total:
            if SHUFFLE: random.shuffle(list_dat)
            batch_nb = 0
        start_idx=batch_nb*batch_size

        Image_data=np.zeros((batch_size,crop_size,crop_size,3),dtype=np.float32)
        Label_gen=np.zeros((batch_size,2), dtype=np.float32)
        if TRAINING_PHASE:
            Label_age=np.zeros((batch_size,101), dtype=np.float32)
        else:
            True_age_mean=np.zeros(batch_size, dtype=np.float32)
            True_age_std =np.zeros(batch_size, dtype=np.float32)

        for k in xrange(batch_size):
            dat = list_dat[start_idx+k]  
            if RANDOM_CROP:
                x_off=np.random.randint(w_max)
                y_off=np.random.randint(h_max)
            else:
                x_off=w_max_half
                y_off=h_max_half
            img=image.img_to_array(Image.open(img_src+dat['add']).convert('RGB').crop(dat['rect']).resize( (w0,h0)))
            Image_data[k] = img[y_off:y_off+crop_size, x_off:x_off+crop_size,:]
            if TRAINING_PHASE:
                Label_age[k][dat['age']]=1
            else:
                True_age_mean[k]=dat['age']
                True_age_std[k]=dat['age_std']

        Image_data= preprocess_input(Image_data)
        #yield ({'input': Image_data}, {'output_gender': Label_gen, 'output_age': Label_age})
        if TRAINING_PHASE:
            targets={'age': Label_age}
        else:
            targets={'age': True_age_mean, 'age_std': True_age_std}
        yield [Image_data, targets]

def de_preprocess_image(image_demeaned):
    return (image_demeaned+[103.939, 116.779, 123.68])[:,:,::-1].astype(np.uint8)


def show_image_label(Image_data, Labels):
    gender_list=['F', 'M']
    from pylab import *
    for k in xrange(Image_data.shape[0]):
        print k,Image_data.shape[0]
        figure(1)
        imshow(de_preprocess_image(Image_data[k]))
        title('Age: %d'% (Labels['age'][k].argmax()))
        plt.show()


if __name__ == '__main__':
    list_dat = get_train_list()
    dat_gen = generate_data(list_dat, 4, 128, 128, 112 , SHUFFLE=True, RANDOM_CROP=True)
    #dat_gen = generate_data(list_dat, 4, 256, 256, 224 , SHUFFLE=True, RANDOM_CROP=False)
    Image_data, Labels=dat_gen.next()
    show_image_label(Image_data, Labels)
