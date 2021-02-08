#%%
import tensorflow as tf
import numpy as np

## basic

x= np.arange(12)
print(x)
print(x.shape)
print(x.size)
# %%
X = x.reshape(3,4)
print(X)
# %%
print(np.zeros((2,3,4)))
print(np.ones((2,3,4)))
# %%
# np.random.normal(mean,std,size)
print(np.random.normal(0,1,size=(3,4)))
# %%

##Operation

x=np.array([1,2,4,8])
y=np.array([2,2,2,2])

print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x**y)
print(np.exp(x))
# %%

X=np.arange(12).reshape(3,4)
Y=np.array([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
print("axis=0",np.concatenate([X,Y],axis=0))
print("axis=1",np.concatenate([X,Y],axis=1))
# %%
X==Y
# %%
X.sum()
# %%
# Broadcasting Mechanism
a=np.arange(3).reshape(3,1)
b=np.arange(2).reshape(1,2)
print(a,b)
print(a+b) ## Broadcast
# %%
# Indexing & Slicing
X[-1],X[1:3]
# %%
X[0:2,:]
# %%
X[0:2,:]=12
X
