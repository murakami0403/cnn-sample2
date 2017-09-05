# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.models import model_from_json
from keras.utils import np_utils

import glob
import numpy as np
import os.path
from pandas import DataFrame
from pandas import read_csv
from PIL import Image
import matplotlib.pyplot as plt
import time

def load_imaegs(image_list,label_list,data_type):
        
    # ./data/train 以下のorange,appleディレクトリ以下の画像を読み込む。
    for dir in os.listdir("./hand_or_not/"+data_type):
        if dir == ".DS_Store":
            continue
    
        dir1 = "./hand_or_not/"+data_type+"/" + dir 
        label = 0
    
        if dir == "hands":    # handsはラベル0
            label = 0
        elif dir == "nothands": # nothandsはラベル1
            label = 1
    
        for file in os.listdir(dir1):
            if file != ".DS_Store":
                # 配列label_listに正解ラベルを追加(dog:0 cat:1)
                label_list.append(label)
                filepath = dir1 + "/" + file
                # 画像を25x25pixelに変換し、1要素が[R,G,B]3要素を含む配列の25x25の２次元配列として読み込む。
                # [R,G,B]はそれぞれが0-255の配列。
                image = np.array(Image.open(filepath).resize((25, 25)))
                #print(filepath)
                # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
                #image = image.transpose(2, 0, 1)
                # 出来上がった配列をimage_listに追加。
                image_list.append(image / 255.)
    
    # kerasに渡すためにnumpy配列に変換。
    image_list = np.array(image_list)

def load_model(model_dir,nametag):
    model_filename = 'cnn_model_hands'+str(nametag)+'.json'
    weights_filename = 'cnn_model_weights_hands'+str(nametag)+'.hdf5'
    json_string = open(os.path.join(model_dir, model_filename)).read()
    model = model_from_json(json_string)
    model.load_weights(os.path.join(model_dir,weights_filename))
    model.compile(loss="categorical_crossentropy", optimizer="adadelta",
                  metrics=["accuracy"])
    return model

def build_model():
        
    # input image dimensions
    img_rows, img_cols = 25, 25
    # number of convolutional filters to use
    nb_filters = 20
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 5
    
    model = Sequential()
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                            padding="valid",
                            input_shape=(img_rows, img_cols, 3)))
    model.add(Activation("relu"))
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adadelta",
                  metrics=["accuracy"])
    return model

def predict_label(model,filepath):
    image = np.array(Image.open(filepath).resize((25, 25)))/255
    expanded = np.expand_dims(image, axis=0)
    predict_class = model.predict_proba(expanded, batch_size=32)
    print(predict_class)

def save_model(model,model_dir,epoch):
    print('save the architecture of a model')
    json_string = model.to_json()
    open(os.path.join(model_dir,'cnn_model_hands'+str(epoch)+'.json'), 'w').write(json_string)
    yaml_string = model.to_yaml()
    open(os.path.join(model_dir,'cnn_model_hands'+str(epoch)+'.yaml'), 'w').write(yaml_string)
    print('save weights')
    model.save_weights(os.path.join(model_dir,'cnn_model_weights_hands'+str(epoch)+'.hdf5'))
    
def plot_history(history):
    # 精度の履歴をプロット
    plt.plot(history.history['acc'],"o-",label="accuracy")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('./acc.png')

    # 損失の履歴をプロット
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig('./loss.png')

# 学習用のデータを作る.
image_list = []
label_list = []
test_image_list = []
test_label_list = []

load_imaegs(image_list,label_list,'train')
load_imaegs(test_image_list,test_label_list,'test')

batch_size = 100
nb_classes = 2
nb_epoch = 100
f_log = './log'
f_model = './model'

# the data, shuffled and split between tran and test sets
x_train = np.array(image_list)
x_test = np.array(test_image_list)
y_train = np.array(label_list)
y_test = np.array(test_label_list)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

#loading model
if(1):        
    model = load_model(f_model,nb_epoch)
else:
    model = build_model()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1)
    # 学習履歴をプロット
    plot_history(history)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test score:", score[0])
print("Test accuracy:", score[1])

save_model(model,f_model,nb_epoch)

#predict_class = model.predict_proba(x_test[:], batch_size=32)
#print(predict_class)


'''
for f in glob.glob('./hand_or_not/test/*/*'):
    if(f[-3:] not in ['png','jpg']):
        continue
    print('**********************************************************')
    print(f)
    start = time.time()
    predict_label(model,f)
        
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
'''