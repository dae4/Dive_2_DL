#%%
import numpy as np
from mpl_toolkits import mplot3d
import torch
from func import *
##Convex Functions
f = lambda x: 0.5 * x**2  # Convex
g = lambda x: torch.cos(np.pi * x)  # Nonconvex
h = lambda x: torch.exp(0.5 * x)  # Convex

x, segment = torch.arange(-2, 2, 0.01), torch.tensor([-1.5, 1])
use_svg_display()
_, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    plot([x, segment], [func(x), func(segment)], axes=ax)
# %%
## Properties
f = lambda x: (x - 1) ** 2
set_figsize()
plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
# %%
