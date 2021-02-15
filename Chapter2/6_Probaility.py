#%%
import numpy as np
## probability * number of case

fair_probs = [1.0 / 6] * 6 
# np.random.multinomial(number of try, probability)
np.random.multinomial(1, fair_probs)
# %%
np.random.multinomial(10, fair_probs)
# %%
import matplotlib.pyplot as plt
counts = np.random.multinomial(10, fair_probs, size=500)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
#cumsum 누적합

estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)
for i in range(6):
    plt.plot(estimates[:, i],label=("P(die=" + str(i + 1) + ")"))
    plt.axhline(y=0.167, color='black', linestyle='dashed')
    plt.gca().set_xlabel('Groups of experiments')
    plt.gca().set_ylabel('Estimated probability')
    plt.legend();
## 결론 : n이 많아질수록 거의 확률이 비슷