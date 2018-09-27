import matplotlib.pyplot as plt
import numpy as np
from scipy import misc, ndimage
import keras
from keras.preprocessing.image import ImageDataGenerator

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0, 2, 3, 1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims) // rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    zoom_range=0.1,
    channel_shift_range=10,
    horizontal_flip=True)

image_path='cats-and-dogs/train/dog/137.jpg'

image = np.expand_dims(ndimage.imread(image_path), 0)

plt.imshow(image[0])

aug_iter = gen.flow(image)

aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]

plots(aug_images, figsize=(20, 7), rows=2)