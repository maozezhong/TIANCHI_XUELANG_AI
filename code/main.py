# coding=utf-8
#################################################
# finetune
#################################################
import pandas as pd
import numpy as np
import os
import imageio
import random
import cv2

from PIL import Image as pil_image
from skimage.transform import resize as imresize
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import initializers
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Maximum
from keras.layers import ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import concatenate
from keras import regularizers
from keras.layers import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU

from keras.applications.densenet import DenseNet169
from keras.applications.densenet import preprocess_input

#######################################
# 在训练的时候置为1
from keras import backend as K
K.set_learning_phase(1)
#######################################

EPOCHS = 50
RANDOM_STATE = 2018
learning_rate = 0.003

TRAIN_DIR = '../data/data_for_train'
VALID_DIR = '../data/data_for_valid'    

def get_callbacks(filepath, patience=2):
    lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=1e-5, patience=patience, verbose=1, min_lr = 0.00001)
    msave = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True) #该回调函数将在每个epoch后保存模型到filepath
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=patience*3+2, verbose=1, mode='auto')
    return [lr_reduce, msave, earlystop]

def add_new_last_layer(base_model, nb_classes, drop_rate=0.):
    """Add last layer to the convnet
    Args:
        base_model: keras model excluding top
        nb_classes: # of classes
    Returns:
        new keras model with last layer
    """
    x = base_model.output
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x) 
    predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return model

# 能一定程度上防止过拟合的交叉熵
def mycrossentropy(e = 0.1,nb_classes=2):
    '''
        https://spaces.ac.cn/archives/4493
    '''
    def mycrossentropy_fixed(y_true, y_pred):
        return (1-e)*K.categorical_crossentropy(y_true,y_pred) + e*K.categorical_crossentropy(K.ones_like(y_pred)/nb_classes, y_pred)
    return mycrossentropy_fixed

def get_model():
    '''
    获得模型
    '''

    base_model = DenseNet169(include_top=False, weights='imagenet', input_shape=(IN_WIDTH,INT_HEIGHT, 3))

    model = add_new_last_layer(base_model, 2)
    model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9), loss=[mycrossentropy()], metrics=['accuracy'])
    model.summary()

    return model

def train_model(save_model_path, BATCH_SIZE, IN_SIZE):

    IN_WIDTH = IN_SIZE
    INT_HEIGHT = IN_SIZE

    callbacks = get_callbacks(filepath=save_model_path, patience=3)
    model = get_model()

    train_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,  
        horizontal_flip = True, 
        vertical_flip = True,
        rotation_range=30,
        shear_range=0.1
    )

    valid_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        directory = TRAIN_DIR, 
        target_size = (IN_WIDTH, INT_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=2018,
        interpolation='antialias',  # PIL默认插值下采样的时候会模糊
    )

    valid_generator = valid_datagen.flow_from_directory(
        directory = VALID_DIR,
        target_size = (IN_WIDTH, INT_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=2018,
        interpolation='antialias',  # PIL默认插值下采样的时候会模糊
    )

    model.fit_generator(
        train_generator, 
        steps_per_epoch = 1*(train_generator.samples // BATCH_SIZE + 1), 
        epochs = EPOCHS,
        max_queue_size = 1000,
        workers = 1,
        verbose = 1,
        validation_data = valid_generator, #valid_generator,
        validation_steps = valid_generator.samples // BATCH_SIZE, #valid_generator.samples // BATCH_SIZE + 1, #len(valid_datagen)+1, 
        callbacks = callbacks
        )

def predict(weights_path, IN_SIZE):
    '''
    对测试数据进行预测
    '''

    IN_WIDTH = IN_SIZE
    INT_HEIGHT = IN_SIZE
    K.set_learning_phase(0)

    test_pic_root_path = '../data/xuelang_round1_test_b'

    filename = []
    probability = []

    #1#得到模型
    model = get_model()
    model.load_weights(weights_path)

    #2#预测
    for parent,_,files in os.walk(test_pic_root_path):
        for line in files:

            pic_path = os.path.join(test_pic_root_path, line.strip())

            ori_img = cv2.imread(pic_path)

            img = load_img(pic_path, target_size=(INT_HEIGHT, IN_WIDTH, 3), interpolation='antialias')
            img = img_to_array(img)
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img)[0] 
            index = list(prediction).index(np.max(prediction))
            pro = list(prediction)[0]   #bad对应的概率大小,因为bad是在index为0的位置
            pro = round(pro, 5)
            if pro == 0:
                pro = 0.00001
            if pro == 1:
                pro = 0.99999

            #存入list
            filename.append(line.strip())
            probability.append(pro)
            
    #3#写入csv
    res_path = "../submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"
    dataframe = pd.DataFrame({'filename': filename, 'probability': probability})
    dataframe.to_csv(res_path, index=False, header=True)

def main():

    #～～～～～～～～～～～～～～～～～  数据处理部分  ～～～～～～～～～～～～～～～～～
    #0# 解压缩原始数据文件
    # 抱歉，有密码，需要手动解压

    #1# 数据转换
    ##1.1## 将图片分为good和bad两类
    os.system("python split_good_bad.py")
    ##1.2## 将所有xml文件提取到data/xml文件夹下面
    os.system("python extract_xml.py")

    #2# 线下训练数据增强
    os.system("python DataAugmentForTrain.py")
    
    #3# 线下增强数据作为validaiton数据
    os.system("python DataAugmentForValid.py")

    #4# 讲所有训练图片copy到 data/data_for_train文件夹下面
    os.system("python del_copy_for_train.py")

    #～～～～～～～～～～～～～～～～～  模型训练及预测part1  ～～～～～～～～～～～～～～～～～
    weights_path = './model_weight1.hdf5'
    # train_model(save_model_path=weights_path, BATCH_SIZE=6, IN_SIZE=500)    # 线上0.9157
    predict(weights_path=weights_path, IN_SIZE=500)

    weights_path = './model_weight2.hdf5'
    # train_model(save_model_path=weights_path, BATCH_SIZE=4, IN_SIZE=550)    # 线上0.918
    predict(weights_path=weights_path, IN_SIZE=550)

    weights_path = './model_weight3.hdf5'
    # train_model(save_model_path=weights_path, BATCH_SIZE=4, IN_SIZE=600)    # 线上0.914
    predict(weights_path=weights_path, IN_SIZE=600)

    # #～～～～～～～～～～～～～～～～～  模型训练及预测part2  ～～～～～～～～～～～～～～～～～
    # # 注意，由于团队是分两条路做的，part2部分是用的pytorch，数据需要手动处理一下，具体见README.md
    # os.system("mv ../data/data_split/good ../data/data_split/0")
    # os.system("mv ../data/data_split/bad ../data/data_split/1")
    # os.system("python xuelangzsp.py")

    #～～～～～～～～～～～～～～～～～ 融合模型得到最终的结果 ～～～～～～～～～～～～～～～～～
    os.system("python merge.py")

if __name__=='__main__':
    main()
