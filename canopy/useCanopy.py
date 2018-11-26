# -*- coding: utf-8 -*-
# @Author: Alan Lau
# @Date: 2017-09-06 17:47:51
# @Last Modified by:   Alan Lau
# @Last Modified time: 2017-09-06 17:47:51


from canopy import Canopy
import numpy as np
from PIL import Image


# dataset = np.random.rand(10, 3)
dataset = Image.open('1.jpg')
image = np.array(dataset, dtype=np.float64) / 255
# print(image)
print(image.shape)
# print(dataset)
gc = Canopy(image)
gc.setThreshold(0.6, 0.4)
canopies = gc.clustering()
print(len(canopies))
