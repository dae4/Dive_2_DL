import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from func import *


pretrained_net = torchvision.models.resnet18(pretrained=True)
pretrained_net.layer4[1], pretrained_net.avgpool, pretrained_net.fc