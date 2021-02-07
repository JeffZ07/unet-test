import scipy.misc as sm
import numpy as np
import imageio
import skimage.transform as trans
import skimage.io as io
from skimage.io import imsave, imread
from matplotlib import pyplot as plt
import cv2

import os


def load_hv_images(path):
    paths = path.split(';')
    root = "data/Jeff/rawinputs/"

    image = io.imread(root+paths[0], as_gray=False)
    # label = sm.imread(paths[1], mode='L')
    column = io.imread(root+paths[2], as_gray=True)
    wall = io.imread(root+paths[3], as_gray=True)
    beam = io.imread(root+paths[4], as_gray=True)
    grid = io.imread(root+paths[5], as_gray=True)

    image = trans.resize(image, (512, 512, 3), mode='constant')

    wall = trans.resize(wall, (512, 512))
    column = trans.resize(column, (512, 512))
    beam = trans.resize(beam, (512, 512))
    grid = trans.resize(grid, (512, 512))

    # wall = trans.resize(wall, (512, 512)) / 255.
    # column = trans.resize(column, (512, 512)) / 255.
    # beam = trans.resize(beam, (512, 512)) / 255.
    # grid = trans.resize(grid, (512, 512)) / 255.

    c_ind = (column < 0.5).astype(np.uint8)
    w_ind = (wall < 0.5).astype(np.uint8)
    b_ind = (beam < 0.5).astype(np.uint8)
    g_ind = (grid < 0.5).astype(np.uint8)

    label = np.zeros((512, 512))
    label[c_ind == 1] = 1  # is column
    label[w_ind == 1] = 2  # is wall
    label[b_ind == 1] = 3  # is beam
    label[g_ind == 1] = 4  # is gird


    label = label.astype(np.uint8)
    label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_CUBIC)
    img_new = np.asarray([label / 255.])
    outdir = "data/Jeff/trainannot/"+paths[0]

    imsave(outdir, label)
    # cv2.imwrite(outdir, img_new)

def LoadAllinOne2():
    root = "data/Jeff/rawinputs/"

    files = os.listdir(root)

    for file in files:
        image = cv2.imread(root + file)
        label = np.zeros((image.shape[0], image.shape[1],4))
        # image = trans.resize(image, (512, 512, 3), mode='constant')
        for j in range(0, image.shape[0]):
            for i in range(0, image.shape[1]):
                item = image[j][i]
                if isWall(item):
                    label[j][i][2] = 1
                elif isBeam(item):
                    label[j][i][3] = 1
                elif isColumn(item):
                    label[j][i][1] = 1
                else:
                    label[j][i][0] = 1

        merge = ind2rgb2(label, color_map=plan_map)
        plt.imshow(merge)
        plt.show()
        filestring = file.split()
        outdir = "data/Jeff/trainannot/" +filestring[0] +filestring[1]+filestring[2] +" - label.png"
        imsave(outdir, label)

def LoadAllinOne():
    root = "data/Jeff/rawinputs/"

    files = os.listdir(root)

    for file in files:
        image = cv2.imread(root + file)
        label = np.zeros((image.shape[0], image.shape[1]))
        # image = trans.resize(image, (512, 512, 3), mode='constant')
        for j in range(0, image.shape[0]):
            for i in range(0, image.shape[1]):
                item = image[j][i]
                if isWall(item):
                    label[j][i] = 2
                elif isBeam(item):
                    label[j][i] = 3
                elif isColumn(item):
                    label[j][i] = 1
                else:
                    label[j][i] = 0

        merge = ind2rgb(label, color_map=plan_map)
        plt.imshow(merge)
        plt.show()
        filestring = file.split()
        outdir = "data/Jeff/trainannot/"+filestring[0]+ " - label.png"
        imsave(outdir, label)

plan_map = {
    0: [255, 255, 255],  # background
    1: [165, 165, 0],  # column
    2: [0, 165, 0],  # wall
    3: [165, 0, 0],  # beam
    4: [0, 0, 255],  # gird
    5: [255, 160, 96],  # not used
    6: [255, 224, 224],  # not used
    7: [224, 224, 224],  # not used
    8: [224, 224, 128]  # not used

}



def ind2rgb2(ind_im):
    rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

    for j in range(0, ind_im.shape[0]):
        for i in range(0, ind_im.shape[1]):
            item = ind_im[j][i]
            if item[1] == 1:
                rgb_im[j][i] = [165, 165, 0]
            elif item[2] == 1:
                rgb_im[j][i] = [0, 165, 0]
            elif item[3] == 1:
                rgb_im[j][i] = [165, 0, 0]
            else:
                rgb_im[j][i] = [255, 255, 255]

    return rgb_im

def ind2rgb(ind_im, color_map):
    rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

    for i, rgb in color_map.items():
        rgb_im[(ind_im == i)] = rgb

    return rgb_im

def isWall(item):
    aa = item[0]
    bb = item[1]
    cc = item[2]
    #Lines
    if (item[0] == 0 and item[1] == 166 and item[2] == 0):
        return True
    else:
        return False

def isBeam(item):
    aa = item[0]
    bb = item[1]
    cc = item[2]
    #Lines
    if (item[0] == 250 and item[1] == 250 and item[2] == 0):
        return True
    else:
        return False

def isColumn(item):
    aa = item[0]
    bb = item[1]
    cc = item[2]
    #Lines
    if (item[0] == 0 and item[1] == 250 and item[2] == 250):
        return True
    else:
        return False

if __name__ == '__main__':
    LoadAllinOne2()