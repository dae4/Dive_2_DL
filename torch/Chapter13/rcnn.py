#%%
import torch
import torchvision

X = torch.arange(16.).reshape(1, 1, 4, 4)
X
# %%
rois = torch.Tensor([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
# %%
