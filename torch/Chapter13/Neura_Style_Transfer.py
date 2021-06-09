from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import pyplotlib.mas plt
import pillow.Imageas Image
from func import *

set_figsize()
content_img = Image.open('../img/rainier.jpg')
plt.imshow(content_img)

style_img = Image.open('../img/autumn-oak.jpg')
plt.imshow(style_img)