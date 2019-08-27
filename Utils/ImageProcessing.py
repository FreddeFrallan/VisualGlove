from PIL import Image
import numpy as np


def images2Numpy(imgs):
    return [np.asarray(i) for i in imgs]


def resizeImages(imgs, size):
    return [i.resize(size) for i in imgs]


def removeAlphaChannel(img, color=(255, 255, 255)):
    img.load()  # needed for split()
    background = Image.new('RGB', img.size, color)
    background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
    return background


def blendImages(imgs, asNumpy=True):
    imgs = np.mean(np.array(imgs), axis=0)
    imgs = np.array(np.round(imgs), dtype=np.uint8)
    return imgs if asNumpy else Image.fromarray(imgs)
