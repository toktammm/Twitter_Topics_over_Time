import numpy as np
import copy
import pickle
import xlrd
from scipy.stats import entropy
import matplotlib.pyplot as plt

'''alpha = 0 -> less dispersity, higher significance, 
alpha = 1 -> high dispersity, low significance'''

alpha = 0
n_b = 26 # number of bins, change to any other number if needed

# path to pickle file of the results
tot_pickle_path = input('enter path to results.pickle:')

tot_pickle = open(tot_pickle_path, 'rb')
par = pickle.load(tot_pickle)

all_disp = []
for i in range(n_t):
    psi = par['psi'][i]
    all_disp.append(pow(entropy(psi, base=2), alpha) * pow((np.log2(len(psi)) - entropy(psi, base=2)), (1-alpha)))


#  SDT plot
# uniform distribution
uni = np.full((n_b), 1./n_b)

dis = copy.deepcopy(uni)
sdt = pow(entropy(dis, base=2), alpha) * pow((np.log2(n_b) - entropy(dis, base=2)), (1-alpha))
print('uniform: ', sdt)


delta = np.zeros(n_b)
delta[np.random.randint(0, n_b-1)] = 1
sdt = pow(entropy(delta, base=2), alpha) * pow((np.log2(n_b) - entropy(delta, base=2)), (1-alpha))
print('Delta: ', sdt)

periodic = np.zeros(n_b)
for num in range(0, n_b):
    if num %2 == 0:
        periodic[num] = 1
periodic = periodic/sum(periodic)
sdt = pow(entropy(periodic, base=2), alpha) * pow((np.log2(n_b) - entropy(periodic, base=2)), (1-alpha))
print('periodic: ', sdt)

# plots
plt.ylim(0, 1)
plt.step(np.linspace(0, n_b, n_b), dis, linewidth=2, color='green')
plt.grid(axis='x', color='0.95')
plt.grid(axis='y', color='0.95')
plt.legend(['uniform'])
plt.show()


# SDT_entropy range:
h_range = np.linspace(0, np.log2(n_b), 100)
sdtls = []
for i in h_range:
    sdtls.append(pow(i, alpha) * pow((np.log2(n_b) - i), (1-alpha)))
plt.ylim(0, 3)
plt.plot(np.array(h_range), np.array(sdtls), linewidth=2, color='navy')
plt.grid(axis='x', color='0.95')
plt.grid(axis='y', color='0.95')
plt.xlabel('entropy')
plt.ylabel('SDT')
plt.show()

