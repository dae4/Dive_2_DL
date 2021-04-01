#%%
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') # Column names
    f.write('NA,Pave,127500\n') # Each row represents a data example
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
# %%
import pandas as pd

data = pd.read_csv(data_file)
print(data)
# %%
## Handling Missing Data
inputs, outpus = data.iloc[:,0:2],data.iloc[:,0:2]
print(inputs)
inputs = inputs.fillna(inputs.mean())
print(inputs)
# %%
inputs =pd.get_dummies(inputs,dummy_na=True)
print(inputs)
# %%
