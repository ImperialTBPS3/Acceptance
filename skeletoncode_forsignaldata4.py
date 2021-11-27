# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit


df = pd.read_pickle('data/total_dataset.pkl')
df = pd.read_pickle('data/sig.pkl')

q2 = df['q2']

bin_ranges = [[0.1, 0.98],
              [1.1, 2.5],
              [2.5, 4.0],
              [4.0, 6.0],
              [6.0, 8.0],
              [15.0,17.0],
              [17.0,19.0],
              [11.0, 12.5],
              [1.0,6.0],
              [15.0,17.9]]

n = len(bin_ranges)

bins = [[] for i in range(n)]

for i in range(n):
    bins[i] = df[(df['q2'] > bin_ranges[i][0]) & (df['q2'] < bin_ranges[i][1])]
    

    
#%%

bins[0].head()

plt.hist(bins[3]['costhetal'], bins=25, density=True)
plt.xlabel(r'$cos(\theta_l)$')
plt.ylabel(r'Number of candidates')
plt.grid()
plt.show()

#%%

df = pd.read_pickle('data/total_dataset.pkl')
df = pd.read_pickle('data/sig.pkl')
df = pd.read_pickle('data/acceptance_mc.pkl')
'''
ctl = df['costhetal']
ctk = df['costhetak']
phi = df['phi']
q2 = df['q2']


plt.figure()
n, bins1, patches = plt.hist(ctl, bins=25, density=True)
bin_center = bins1[:-1] + np.diff(bins1) / 2
fit_ctl = np.polynomial.legendre.Legendre.fit(bin_center, n, 4)

plt.figure()
n, bins1, patches = plt.hist(ctk, bins=25, density=True)
bin_center = bins1[:-1] + np.diff(bins1) / 2
fit_ctk = np.polynomial.legendre.Legendre.fit(bin_center, n, 5)

plt.figure()
n, bins1, patches = plt.hist(phi, bins=25, density=True)
bin_center = bins1[:-1] + np.diff(bins1) / 2
fit_phi = np.polynomial.legendre.Legendre.fit(bin_center, n, 6)

plt.figure()
n, bins1, patches = plt.hist(q2, bins=25, density=True)
bin_center = bins1[:-1] + np.diff(bins1) / 2
fit_q2 = np.polynomial.legendre.Legendre.fit(bin_center, n, 5)
'''



#%%

# creating bin ranges

'''
bin ranges:

0.1 < q2 < 1 : 0.1
1.0 < q2 < 6.0 : 0.2
6.0 < q2 < 22.0 : 0.5
'''

legendre_bins = []
range1 = np.arange(0, 1, 0.1)
range2 = np.arange(1, 6, 0.2)
range3 = np.arange(6, 20, 0.5)
for i in range1:
    legendre_bins.append(i)
for i in range2:
    legendre_bins.append(i)
for i in range3:
    legendre_bins.append(i)

#%%

# creating arrays of angles in separate q2 bins

bins_ctl = [[] for i in range(len(legendre_bins) - 1)]
bins_ctk = [[] for i in range(len(legendre_bins) - 1)]
bins_phi = [[] for i in range(len(legendre_bins) - 1)]

for i in range(len(legendre_bins) - 1):
    row = df[(df['q2'] > legendre_bins[i]) & (df['q2'] < legendre_bins[i+1])]
    
    row_ctl = row['costhetal']
    bins_ctl[i] = row_ctl
    
    row_ctk = row['costhetak']
    bins_ctk[i] = row_ctk
    
    row_phi = row['phi']
    bins_phi[i] = row_phi


#%%

# creating fit for each q2 bin

fits_ctl = [[] for i in range(len(legendre_bins) - 1)]
fits_ctk = [[] for i in range(len(legendre_bins) - 1)]
fits_phi = [[] for i in range(len(legendre_bins) - 1)]

for i in range(len(legendre_bins) - 1):
    plt.figure()
    n, bins1, patches = plt.hist(bins_ctl[i], bins=25, density=True)
    bin_center = bins1[:-1] + np.diff(bins1) / 2
    fits_ctl[i] = np.polynomial.legendre.Legendre.fit(bin_center, n, 4)
    plt.close()
    
    plt.figure()
    n, bins1, patches = plt.hist(bins_ctk[i], bins=25, density=True)
    bin_center = bins1[:-1] + np.diff(bins1) / 2
    fits_ctk[i] = np.polynomial.legendre.Legendre.fit(bin_center, n, 5)
    plt.close()

    plt.figure()
    n, bins1, patches = plt.hist(bins_phi[i], bins=25, density=True)
    bin_center = bins1[:-1] + np.diff(bins1) / 2
    fits_phi[i] = np.polynomial.legendre.Legendre.fit(bin_center, n, 6)
    plt.close()
    
    
#%%

# calculating acceptance function for each analysis q2 bin 

ctl = df['costhetal']
ctk = df['costhetak']
phi = df['phi']
q2 = df['q2']

'''
- for each value of costhetal, find its corresponding q2 value and thus which legendre q2 bin it falls into
- evaluate the appropriate legendre fit at that value of costhetal to determine the acceptance function 
- apply normalization
'''

efficiency = []
for i in range(len(ctl)):
    for j in range(len(legendre_bins) - 1):
        if (df['q2'][i] > legendre_bins[j]) & (df['q2'][i] < legendre_bins[j+1]):
            e = fits_ctl[j](ctl[i]) * fits_ctk[j](ctk[i]) * fits_phi[j](phi[i])
            efficiency.append(e)
            break
average = sum(efficiency) / len(efficiency)
efficiency /= average
acceptance = 1 / efficiency

'''
- for each standard model q2 bin, find the total dataset index for each value of costhetal
- match each value of the acceptance function into the appropriate bin
'''

indices = [[] for i in range(len(bins))]
for i in range(len(bins)):
    index = bins[i].index
    indices[i] = index
    
acceptances = [[] for i in range(len(bins))]
for i in range(len(bins)):
    for j in indices[i]:
        a = acceptance[j]
        acceptances[i].append(a)


#%%


def d2gamma_p_d2q2_dcostheta(fl, afb, cos_theta_l, cos_theta_k, phi, bin_number):
    """
    Returns the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param cos_theta_l: cos(theta_l)
    :return:
    """
    ctl = cos_theta_l
    ctk = cos_theta_k
    phi = phi
    c2tl = 2 * ctl ** 2 - 1
    #acceptance = fit_ctl(ctl) 
    #acceptance = fit_ctl(ctl) * fit_ctk(ctk) * fit_phi(phi)
    #average = sum(acceptance) / len(acceptance)
    #acceptance /= average
    #acceptance = 1 / fit_ctl(ctl) 
    #average = sum(acceptance) / len(acceptance)
    #acceptance *= average
    
    '''
    efficiency = []
    for i in range(len(ctl)):
        for j in range(len(legendre_bins) - 1):
            if (df['q2'][i] > legendre_bins[j]) & (df['q2'][i] < legendre_bins[j+1]):
                e = fits_ctl[j](ctl[i]) * fits_ctk[j](ctk[i]) * fits_phi[j](phi[i])
                efficiency.append(e)
                break
    average = sum(efficiency) / len(efficiency)
    efficiency /= average
    acceptance = 1 / efficiency
    '''
                
    # acceptance = 2
    
    acceptance = acceptances[bin_number]

    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) + 8/3 * afb * ctl) * acceptance
    normalised_scalar_array = scalar_array 
    
    #normalised_scalar_array = normalised_scalar_array[0]
    return normalised_scalar_array

def log_likelihood(fl, afb, _bin):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    bin_number = int(_bin)
    _bin = bins[int(_bin)]
    ctl = _bin['costhetal']
    ctk = _bin['costhetak']
    phi = _bin['phi']
    normalised_scalar_array = d2gamma_p_d2q2_dcostheta(fl=fl, afb=afb, cos_theta_l=ctl, cos_theta_k=ctk, phi=phi, bin_number=bin_number)
    return - np.sum(np.log(normalised_scalar_array))

#%%


#%%

_test_bin = 1
_test_afb = 0.7
_test_fl = 0.0

x = np.linspace(-1, 1, 500)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
ax1.plot(x, [log_likelihood(fl=i, afb=_test_afb, _bin=_test_bin) for i in x])
ax1.set_title(r'$A_{FB}$ = ' + str(_test_afb))
ax1.set_xlabel(r'$F_L$')
ax1.set_ylabel(r'$-\mathcal{L}$')
ax1.grid()
ax2.plot(x, [log_likelihood(fl=_test_fl, afb=i, _bin=_test_bin) for i in x])
ax2.set_title(r'$F_{L}$ = ' + str(_test_fl))
ax2.set_xlabel(r'$A_{FB}$')
ax2.set_ylabel(r'$-\mathcal{L}$')
ax2.grid()
plt.tight_layout()
plt.show()

#%%

bin_number_to_check = 0  # bin that we want to check in more details in the next cell
bin_results_to_check = None

log_likelihood.errordef = Minuit.LIKELIHOOD
decimal_places = 3
starting_point = [-0.1,0.0]
fls, fl_errs = [], []
afbs, afb_errs = [], []
for i in range(len(bins)):
    m = Minuit(log_likelihood, fl=starting_point[0], afb=starting_point[1], _bin=i)
    m.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
    m.limits=((-1.0, 1.0), (-1.0, 1.0), None)
    m.migrad()
    m.hesse()
    if i == bin_number_to_check:
        bin_results_to_check = m
    fls.append(m.values[0])
    afbs.append(m.values[1])
    fl_errs.append(m.errors[0])
    afb_errs.append(m.errors[1])
    print(f"Bin {i}: {np.round(fls[i], decimal_places)} pm {np.round(fl_errs[i], decimal_places)},", f"{np.round(afbs[i], decimal_places)} pm {np.round(afb_errs[i], decimal_places)}. Function minimum considered valid: {m.fmin.is_valid}")
    
    
#%%


plt.figure(figsize=(8, 5))
plt.subplot(221)
bin_results_to_check.draw_mnprofile('afb', bound=3)
plt.subplot(222)
bin_results_to_check.draw_mnprofile('fl', bound=3)

#%%
'''
bin_to_plot = 3
number_of_bins_in_hist = 25
cos_theta_l_bin = bins[bin_to_plot]['costhetal']
hist, _bins, _ = plt.hist(cos_theta_l_bin, bins=number_of_bins_in_hist)
x = np.linspace(-1, 1, number_of_bins_in_hist)
pdf_multiplier = np.sum(hist) * (np.max(cos_theta_l_bin) - np.min(cos_theta_l_bin)) / number_of_bins_in_hist
y = d2gamma_p_d2q2_dcostheta(fl=fls[bin_to_plot], afb=afbs[bin_to_plot], cos_theta_l=x, cos_theta_k=x, phi=x) * pdf_multiplier
plt.plot(x, y, label=f'Fit for bin {bin_to_plot}')
plt.xlabel(r'$cos(\theta_l)$')
plt.ylabel(r'Number of candidates')
plt.legend()
plt.grid()
plt.show()
plt.tight_layout()
plt.show()
'''
#%%


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
ax1.errorbar(np.linspace(0, len(bins) - 1, len(bins)), fls, yerr=fl_errs, fmt='o', markersize=2, label=r'$F_L$', color='red')
ax2.errorbar(np.linspace(0, len(bins) - 1, len(bins)), afbs, yerr=afb_errs, fmt='o', markersize=2, label=r'$A_{FB}$', color='red')
ax1.grid()
ax2.grid()
ax1.set_ylabel(r'$F_L$')
ax2.set_ylabel(r'$A_{FB}$')
ax1.set_xlabel(r'Bin number')
ax2.set_xlabel(r'Bin number')
plt.tight_layout()
plt.show()