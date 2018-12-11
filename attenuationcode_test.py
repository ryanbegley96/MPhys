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

###############################################################################
#loading in fake galaxy fluxes.
lambda_fake,f_fake1,f_fake2 = np.genfromtxt('ryan_testcurves.csv', \
                                    delimiter=',', skip_header=1, unpack=True)
#load in calzetti and smc curves
dustcurves = np.genfromtxt('dustcurves.csv', delimiter=',', skip_header=1, \
                            unpack=True)
smc_curve = dustcurves[1]
calz_curve = dustcurves[2]
###############################################################################

median_z = clipped_df['ZSPEC_NEW'].median()
#de_z_lambda = band_wavelengths_inter / (1. + median_z)

#assumed intrinsic beta
beta_intrinsic = -2.4

lambda_fake = lambda_fake / 10000. #converting A -> um
#print(lambda_fake)
#want to normalise everything to V=5500A=0.55um
#corresponding to fake gal flux f1=3.09096, f2=0.65759

f_fake1_norm = f_fake1 / 3.09096
f_fake2_norm = f_fake2 / 0.65759
#print(f_fake1_norm)

f_intrins_norm = ( lambda_fake / 0.55 )**beta_intrinsic

# #plot for normalised fake fluxes
# plt.figure()
# plt.plot(lambda_fake,f_fake1_norm, label='fake1')
# plt.plot(lambda_fake,f_fake2_norm, label='fake2')
# plt.plot(lambda_fake,f_intrins_norm, label='intrinsic')
# plt.xlabel(r'$\lambda/microns$')
# plt.ylabel('$f/f_{V}$')
# plt.xlim(xmin=0.1)
# plt.legend(fontsize='small')
# plt.title('Normalised Fake Flux Curve')


###############################################################################

y_data1 = 1.086*np.log(f_intrins_norm/f_fake1_norm)
y_data2 = 1.086*np.log(f_intrins_norm/f_fake2_norm)
y_data_err = np.full(f_intrins_norm.shape, 0.001)


def model_curve(x, a0, a1, Av):
    return a0*x +a1*(x**2.0) - Av

# def chi_squared1(a0,a1,A_598):
#     return np.sum( ( ( y_data1 - model_curve((1./lambda_fake), a0,a1,A_598))/ y_data_err )**2.0 )
#
# def chi_squared2(a0,a1,A_598):
#     return np.sum( ( ( y_data2 - model_curve((1./lambda_fake), a0,a1,A_598))/ y_data_err )**2.0 )

#CREATE MASK TO ISOLATE IA BANDS ie 0.43-0.86um, take deredshifted
IA_mask = (lambda_fake >= 0.125) & (lambda_fake <= 0.253)
#print(IA_mask)

testx = 1.0 / ( lambda_fake[IA_mask] )
testy1 = y_data1[IA_mask]
testy2 = y_data2[IA_mask]

#method not using chisq
# popt1, pcov1 = curve_fit(model_curve, testx, testy1, sigma=y_data_err[IA_mask])
# print('Fitted params a0,a1,A_598 are -'+str(popt1))
# popt2, pcov2 = curve_fit(model_curve, testx, testy2, sigma=y_data_err[IA_mask])
# print('Fitted params a0,a1,A_598 are -'+str(popt2))

popt1, pcov1 = curve_fit(model_curve, (1.0/lambda_fake), y_data1, sigma=y_data_err)
print('Fitted params a0,a1,A_598 are -'+str(popt1))
popt2, pcov2 = curve_fit(model_curve, (1.0/lambda_fake), y_data2, sigma=y_data_err)
print('Fitted params a0,a1,A_598 are -'+str(popt2))

attn_curve_fit1 = ( popt1[0]*(lambda_fake**-1.0) + popt1[1]*(lambda_fake**-2.0) ) / popt1[2]
attn_curve_fit2 = ( popt2[0]*(lambda_fake**-1.0) + popt2[1]*(lambda_fake**-2.0) ) / popt2[2]

y_data1_fit = popt1[0]*(lambda_fake**-1.0)+popt1[1]*(lambda_fake**-2.0)-popt1[2]

#plot for data + fit against 1/lambda for fake 1 only
# plt.figure()
# plt.plot((1./lambda_fake),y_data1,linestyle='None',marker='.',label='Fake1 Data Points')
# plt.plot((1./lambda_fake),y_data1_fit, label='Fit')
#
# #plt.plot(testx,testy1,linestyle='None',marker='+',label='Fitted pts')
# #plt.axvspan(1/0.125, 1/0.253, facecolor='g', alpha=0.5)
# plt.xlabel(r'$\lambda/microns$')
# plt.ylabel('1.086*ln(Fi/Fo)')
# plt.title('Results from Attenuation Fit')
# plt.legend(fontsize='small')


#plot for final Attenuation curve.
plt.figure()
plt.plot(lambda_fake, attn_curve_fit1, linestyle='-', marker=',', label='fake1')
plt.plot(lambda_fake, attn_curve_fit2, linestyle='-', marker=',', label='fake2')

plt.plot(lambda_fake, smc_curve, label='SMC', linestyle='--')
plt.plot(lambda_fake, calz_curve, label='Calzetti', linestyle='--')

#plt.plot(1/testx,attn_curve_fit1[IA_mask], color='k', linestyle='None',marker=',',label='Fitted pts')
plt.axvspan(0.125, 0.253, facecolor='g', alpha=0.5)

plt.xlim(xmin=0.1)
plt.xlabel(r'1/$\lambda / microns$')
plt.ylabel(r'A_{\lambda}/A_{V}')
plt.legend(fontsize='small')
plt.title('Attenuation Curve from curve_fit for fake galaxies')
plt.show()
