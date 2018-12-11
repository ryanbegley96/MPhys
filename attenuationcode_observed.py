import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from iminuit import Minuit
from pprint import pprint
#import median stack data + associated fitting results
from observationcode import *

#load in calzetti and smc curves for comparison
dustcurves = np.genfromtxt('dustcurves.csv', delimiter=',', skip_header=1, \
                            unpack=True)
smc_curve = dustcurves[1]
calz_curve = dustcurves[2]
wl_dustcurve = wl_range = np.linspace(0.1,0.6,len(calz_curve))
##############################################################################

median_z = clipped_df['ZSPEC_NEW'].median()
beta_intrinsic = -2.4

#band_wavelengths(_inter) and median stack + errors in observationcode
#use beta and norm observed to artificially normalise to V=0.55um


#taking only IA bands from observationcode
f_obs_norm = medianstack_inter / 0.0048351 #num from reading graph
f_obs_norm_err = medianstack_err_inter / 0.0048351

f_intr = ( band_wavelengths_inter / 0.55 )**beta_intrinsic

##############################################################################
#make some equally spaced data over a wider range

wl_range = np.linspace(0.1,0.6,50)
f_obs_norm_range = (normfit/0.0048351) * wl_range**betafit
f_intr_range = ( wl_range / 0.55 )**beta_intrinsic
##############################################################################

#plotting flux curves
plt.figure()

plt.plot(wl_range,f_obs_norm_range, color='k', )
plt.plot(wl_range,f_intr_range, color='k')

plt.plot(band_wavelengths_inter, f_obs_norm, linestyle='None', marker='.',\
            label=r'$\beta_{obs}$ = '+str(round(betafit,2)))
plt.plot(band_wavelengths_inter, f_intr, linestyle='None', marker='.', \
            label=r'$\beta_{int}$ = '+str(beta_intrinsic))

plt.title('Flux curves for fitting')
plt.xlabel(r'$\lambda/microns$')
plt.ylabel('Flux')
plt.legend(fontsize='small')
#plt.show()

y_data = 1.086*np.log10(f_intr/f_obs_norm)
y_data_err = (1.086**2.0)*((f_obs_norm_err/f_obs_norm)**2.0 )
print(y_data,y_data_err)

#testing minuit over full range
y_data_range = 1.086*np.log10(f_intr_range/f_obs_norm_range)
y_data_err_range = np.full(y_data_range.shape,y_data_err.mean())

#Fitting Process
##############################################################################

#some functions defined for fitting dust curve
def model_curve(x, a0, a1, Av):
    return a0*x +a1*(x**2.0) - Av

def chi_squared_discrete(a0,a1,Av):
    return np.sum( ( ( y_data - model_curve((1./band_wavelengths_inter),\
                    a0,a1,Av) ) / y_data_err )**2.0 )

def chi_squared_range(a0,a1,Av):
    return np.sum( ( ( y_data_range - model_curve((1./wl_range),\
                    a0,a1,Av) ) / y_data_err_range )**2.0 )
###############################################################################
###THIS STUFF IS WRONG NOT SURE WHY, will try scipy over full and partial wl.
a0_est = 0.5
a1_est = -0.02
Av_est = 1.

attn_inst = Minuit(chi_squared_range, a0=a0_est, a1=a1_est, Av=Av_est, error_a0=0.01,\
            error_a1=0.01, error_Av=0.01, errordef=1, print_level=1)
fval, params = attn_inst.migrad()

a0_fit = params[0]['value']
a0_fit_err = params[0]['error']
a1_fit = params[1]['value']
a1_fit_err = params[1]['error']
Av_fit = params[2]['value']
Av_fit_err = params[2]['error']
params_arr = np.array([a0_fit,a1_fit,Av_fit])
print('Minuit Fitted params a0,a1,Av are- '+str(params_arr))

attn_curve_fit_minuit = (a0_fit*(wl_range**-1.0)+a1_fit*(wl_range**-2.0))/Av_fit


plt.figure()
plt.plot(wl_range, attn_curve_fit_minuit, label=str(params_arr))
plt.plot(wl_dustcurve, smc_curve, label='SMC', linestyle='--')
plt.plot(wl_dustcurve, calz_curve, label='Calzetti', linestyle='--')
plt.title('Minuit Attn curve.')
plt.xlabel(r'$\lambda/microns$')
plt.ylabel(r'$A_{\lambda}/A_{V}$')
plt.legend(fontsize='small')
plt.show()
"""
###############################################################################
#scipy >> doesnt work over IA range only.
# y_data_range = 1.086*np.log10(f_intr_range/f_obs_norm_range)
# y_data_err_range = np.full(y_data_range.shape,y_data_err.mean())

popt,pcov = curve_fit(model_curve, (1./wl_range), y_data_range, \
            sigma=y_data_err_range)
print('Scipy full range Fitted params a0,a1,Av are -'+str(popt))

attn_curve_fit_scipy = (popt[0]*(wl_range**-1.0)+popt[1]*(wl_range**-2.0))/popt[2]

plt.figure()
plt.plot(wl_range, attn_curve_fit_scipy, label=str(popt))
plt.plot(wl_dustcurve, smc_curve, label='SMC', linestyle='--')
plt.plot(wl_dustcurve, calz_curve, label='Calzetti', linestyle='--')
plt.title(r'Scipy Attn curve - over $\lambda$ range')
plt.xlabel(r'$\lambda/microns$')
plt.ylabel(r'$A_{\lambda}/A_{V}$')
plt.legend(fontsize='small')
plt.show()
"""
