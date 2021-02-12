#%%
import numpy as np

x=np.array(3.0)
y=np.array(2.0)
print(x+y,x*y,x/y,x**y)
# %%
x=np.arange(4)
x
# %%
x[3]
# %%
len(x)
# %%
A=np.arange(20).reshape(5,4)
A
# %%
A.T #Transpose
# %%
B=np.array([[1,2,3],[2,0,4],[3,4,5]])
B
# %%
B==B.T
# %%
X=np.arange(24).reshape(2,3,4)
X
# %%
A=np.arange(20).reshape(5,4)
B=A.copy()
print(A,A+B)
# %%
A*B
# %%
a=2
X=np.arange(24).reshape(2,3,4)
a+X,(a*X).shape
# %%
X=np.arange(4)
X,X.sum()
# %%
A.shape , A.sum()
#%%
# 세로열 합
# 0+4+8+12+16 =40
A_sum_axis0=A.sum(axis=0)
A_sum_axis0,A_sum_axis0.shape
# %%
# 가로열 합
# 0+1+2+3=6
A_sum_axis1=A.sum(axis=1)
A_sum_axis1,A_sum_axis1.shape
# %%
A.sum(axis=(0,1)) # == A.sum()
# %%
A.mean(), A.sum()/A.size
# %%
A.mean(axis=0), A.sum(axis=0)/A.shape[0]
# %%
## keepdims 차원 축소 방지
sum_A = A.sum(axis=1,keepdims=True)
sum_A
# %%
A/sum_A
# %%
# 원소 누적합
# 아래열로 갈수록 누적으로 합
A.cumsum(axis=0)
# %%
y=np.ones(4)
x,y,np.dot(x,y)
# %%
A.shape, x.shape,np.dot(A,x)
#5,4     4,1       5,1
# %%
B=np.ones(shape=(4,3))
np.dot(A,B)
## 5,4 * 4,3  => 5,3
# %%
u=np.array([3,-4])
np.linalg.norm(u) ## L2 norm root(3^2+4^2) = 5
# %%
np.abs(u).sum() ## L1 norm 3+4
# %%
