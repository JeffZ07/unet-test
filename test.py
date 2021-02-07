from model_v2 import *
from data import *
import numpy as np
import cv2
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def image_normalized(file_path):
    '''
    tif，size:512*512，gray
    :param dir_path: path to your images directory
    :return:
    '''
    img = cv2.imread(file_path)
    img_shape = img.shape
    image_size = (img_shape[1],img_shape[0])
    img_standard = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    img_new = img_standard
    img_new = np.asarray([img_new / 255.])
    return img_new,image_size

def ProcessResult(ind_im):
    rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

    for j in range(0, ind_im.shape[0]):
        for i in range(0, ind_im.shape[1]):
            item = ind_im[j][i]
            maxindex = np.argmax(item)
            if maxindex == 1:
                rgb_im[j][i] = [165, 165, 0]
            elif maxindex == 2:
                rgb_im[j][i] = [0, 165, 0]
            elif maxindex == 3:
                rgb_im[j][i] = [165, 0, 0]
            else:
                rgb_im[j][i] = [255, 255, 255]

    return rgb_im


if __name__ == '__main__':

    #path to images which aring wating for predicting
    test_path = "data/Jeff/test"

    # save the predict images
    save_path = "data/Jeff/predict"

    dp = data_preprocess(test_path=test_path,save_path=save_path,flag_multi_class=True,num_classes=12)

    #load model
    model = load_model('./model/Jeff_model_v1.hdf5')

    for name in os.listdir(test_path):
        image_path = os.path.join(test_path,name)
        x,img_size = image_normalized(image_path)
        results = model.predict(x)
        results = ProcessResult(results[0])
        dp.saveResult([results[0]],img_size,name.split('.')[0])
