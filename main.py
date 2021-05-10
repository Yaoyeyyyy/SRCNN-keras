

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:46:01 2021

@author: yy
"""

from keras.models import Sequential
from keras.layers import Conv2D, Input, BatchNormalization
from keras import initializers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,TensorBoard
from keras.optimizers import SGD, Adam
from glob import glob
import numpy
import cv2
import numpy as np
from PIL import Image
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn import preprocessing
import os 
def psnr(target, ref):
    # assume RGB image
    target_data = numpy.array(target, dtype=float)
    ref_data = numpy.array(ref, dtype=float)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = np.sqrt(numpy.mean(diff ** 2.))
    return 20 * np.log10(255. / rmse)


def model():

    SRCNN = Sequential()
    SRCNN.add(Conv2D(filters = 128, kernel_size = 9, kernel_initializer=initializers.random_normal(mean=0.0,stddev=0.001),
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    
    SRCNN.add(Conv2D(filters = 64, kernel_size = 1, kernel_initializer=initializers.random_normal(mean=0.0,stddev=0.001),
                     activation='relu', padding='same', use_bias=True))

    SRCNN.add(Conv2D(filters=1, kernel_size = 5, kernel_initializer=initializers.random_normal(mean=0.0,stddev=0.001),
                     activation='linear', padding='valid', use_bias=True))
    sgd = SGD(lr=0.0001) 
    SRCNN.compile(optimizer=sgd, loss='mse')
    return SRCNN


def train(data_path,label_path,val_data_path,val_label_path):
    srcnn_model = model()
    print(srcnn_model.summary())
    print("got net")
    log_dir = "logs/"
    tensorboard = TensorBoard(log_dir=log_dir)
    model_checkpoint = ModelCheckpoint('SRCNN_hair.hdf5', monitor='val_loss',verbose=1, save_best_only=True)# keras保存最好的模型ModelCheckpoint
    
    if os.path.exists('SRCNN_hair.hdf5'):
        srcnn_model.load_weights('SRCNN_hair.hdf5')
        # 若成功加载前面保存的参数，输出下列信息
        print("checkpoint_loaded")
        print("*********************")
    
    callback_lists = [tensorboard,model_checkpoint]

    print('Fitting model...')
    
    data = np.load(data_path)/255.
    label = np.load(label_path)/255.
    val_data = np.load(val_data_path)/255.
    val_label = np.load(val_label_path)/255.
    
    srcnn_model.fit(data, label, batch_size=32, validation_data=(val_data, val_label),
    callbacks=callback_lists, shuffle=True, epochs=10, verbose=1)

    print('predict train data')

def test(data_path,Pic_save,row,col):
    srcnn_model = model()
    srcnn_model.load_weights("SRCNN_hair.hdf5")
    print('predict test data')
    
    file_list=os.listdir(data_path)
    
    print(len(file_list))
    imgdatas = np.ndarray((len(file_list),row,col,1), dtype=np.float32)
    i=0
    for file in file_list:
        file_path=os.path.join(data_path,file)
        #print(file_path)
        img = Image.open(file_path).convert('L')
        img = img_to_array(img) #float 32
        img /= 255.0 # norm
        #img = img/127.5 - 1.0
        np.nan_to_num(img)
        img[img > 1] = 1
        img[img < 0] = 0
        imgdatas[i] = img 
        i = i+1

    preds = srcnn_model.predict(imgdatas,batch_size=32, verbose=1)
    print(preds.shape) 
    length = preds.shape[0]
    print(length)
    for i in range(length):
        img = preds[i]
        imgnewNp = img*255
        #imgnewNp = (img+1)*127.5
        imgnewNp[imgnewNp > 255] = 255
        imgnewNp[imgnewNp < 0] = 0
        imgnew = array_to_img(imgnewNp)
        dirS=Pic_save+'dark'+str(i)+'.tif'
        imgnew.save(dirS)

def evaluation(dirTest,dirDNN,dirGT,psnrDNN,psnrCUBIC):
    fileTestList=os.listdir(dirTest)
    fileDNNList=os.listdir(dirDNN)
    fileGTList=os.listdir(dirGT)
    length = len(fileTestList)
    
    for i in range(length):
        fileGT=os.path.join(dirGT,fileGTList[i])
        fileDNN=os.path.join(dirDNN,fileDNNList[i])
        fileTest=os.path.join(dirTest,fileTestList[i])

        ref = cv2.imread(fileGT, cv2.IMREAD_GRAYSCALE)[6: -6, 6: -6]
        targerDNN = cv2.imread(fileDNN, cv2.IMREAD_GRAYSCALE)
        targerCUBIC = cv2.imread(fileTest, cv2.IMREAD_GRAYSCALE)[6: -6, 6: -6]
        
        psnrSingleDNN = psnr(targerDNN,ref)
        psnrDNN.append(psnrSingleDNN)
        psnrSingleCUBIC = psnr(targerCUBIC,ref)
        psnrCUBIC.append(psnrSingleCUBIC)
    
if __name__ == "__main__":
    dirTrainInput = 'D:/14.Experiment/code/data/data-student/trainInput32.npy'
    dirTrainLabel = 'D:/14.Experiment/code/data/data-student/trainLabel20.npy'
    dirValInput = 'D:/14.Experiment/code/data/data-student/valInput32.npy'
    dirValLabel = 'D:/14.Experiment/code/data/data-student/valLabel20.npy'
    train(dirTrainInput,dirTrainLabel,dirValInput,dirValLabel)
    dirTest = 'D:/14.Experiment/code/data/data-student/test2Input/'
    dirDNN = 'D:/14.Experiment/code/data/data-student/test2DNN/'
    dirGT = 'D:/14.Experiment/code/data/data-student/test2Label/'
    row = 960
    col = 1280
    test(dirTest,dirDNN,row,col)
    psnrDNN=[]
    psnrCUBIC=[]
    evaluation(dirTest,dirDNN,dirGT,psnrDNN,psnrCUBIC)
