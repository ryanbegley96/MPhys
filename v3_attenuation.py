import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from iminuit import Minuit
from pprint import pprint
from scipy.optimize import curve_fit

from v3_observation import *

#print(betafit)
beta_intrinsic = -2.23 #=-2.31 from fcullen paper for the intr sed below
wl_IA_JHK = np.around( 1.0E4 * band_wavelengths[4:] )

#loading in cullen fiby_bpass300bin_z4 intr SED, wl, f_neb, f_star
cullen_sed = np.genfromtxt('intrinsic_seds/fiby_bpass300bin_z4.spectrum', unpack=True)
cullen_wl = cullen_sed[0]
cullen_fneb = cullen_sed[1]
cullen_fstr = cullen_sed[2]

#isolates the specific IA+JHK data points from the cullen intr sed
IA_JHK_mask = np.isin( cullen_wl, wl_IA_JHK )
fc_ia_wl, fc_ia_fneb, fc_ia_fstr = cullen_wl[IA_JHK_mask], cullen_fneb[IA_JHK_mask], cullen_fstr[IA_JHK_mask]

#mask to isolate wavelength range UV>Optimal + apply to each column
wl_mask = (cullen_wl>=1250) & (cullen_wl<=6300)
cullen_wl = cullen_wl[wl_mask]
cullen_fneb = cullen_fneb[wl_mask]
cullen_fstr = cullen_fstr[wl_mask]

# plt.figure(figsize=(8,4))
# plt.plot(cullen_wl,cullen_fstr, color='k', linestyle='-', linewidth=0.5, label='Stellar')
# plt.plot(cullen_wl,cullen_fneb, color='k', linestyle=':', linewidth=0.5, label='w/ Nebular')
#
# plt.plot(fc_ia_wl, fc_ia_fstr, color='r', linestyle='None', marker='+')
# plt.plot(fc_ia_wl, fc_ia_fneb, color='r', linestyle='None', marker='+', label='IA+JHK')
#
# plt.plot(cullen_wl, (1.837691E+6 * 5500**(-beta_intrinsic) * cullen_wl**(beta_intrinsic) ), color='b', linewidth=0.5, label=r'$\beta$= '+str(beta_intrinsic)+' pinned at V')
#
# plt.xlabel(r'$\lambda / \AA$')
# plt.ylabel(r'$log_{10}(f_{\lambda})$')
#
# plt.xlim(xmin=1250,xmax=6300)
# plt.ylim(ymin=cullen_fstr[cullen_wl==6300],ymax=cullen_fstr[cullen_wl==1250])
# plt.yscale('log')
# plt.legend(fontsize='small')
# plt.show()
###############################################################################
#Fitting process below

f_obs = medianstack[4:]
#using 5% error for now although look into medianstack 3sigma error in future
f_obs_err = 0.05 * medianstack[4:]

#for f_intr, try both fcullen intr SED and just power law
f_intr = fc_ia_wl**beta_intrinsic

y_data1 = 1.086*np.log(f_obs/f_intr)
y_data2 = 1.086*np.log(f_obs/fc_ia_fstr) #using the stellar comp only sed
y_data3 = 1.086*np.log(f_obs/fc_ia_fneb) #using the stellar comp only sed

y_data_err = 1.086*0.43429*(f_obs_err/f_obs) #assuming fintr has no error


def model_curve(x, logk, a0, a1):
    return logk - 0.4 * ( a0 * x**(-1.0) + a1 * x**(-2.0) )

def chi_squared1(logk, a0, a1):
    return np.sum( ( ( y_data1 - model_curve(fc_ia_wl*1.0E-4, logk, a0, a1) ) / y_data_err )**2.0 )

def chi_squared2(logk, a0, a1):
    return np.sum( ( ( y_data2 - model_curve(fc_ia_wl*1.0E-4, logk, a0, a1) ) / y_data_err )**2.0 )

#remove the H band from nebular emission sed as large feature
contam_mask = np.ones(fc_ia_wl.shape, dtype=bool)
contam_mask[17] = 0 #position of H band in wl array
fc_ia_wl_noH, y_data3_noH, y_data_err_noH = fc_ia_wl[contam_mask], y_data3[contam_mask], y_data_err[contam_mask]
def chi_squared3(logk, a0, a1):
    return np.sum( ( ( y_data3_noH - model_curve(fc_ia_wl_noH*1.0E-4, logk, a0, a1) ) / y_data_err_noH )**2.0 )

logk_est = 2.0
a0_est = 0.5
a1_est = -0.1

#round1 w/ basic f_intr power law
attn_inst1 = Minuit(chi_squared1, logk=logk_est, a0=a0_est, a1=a1_est, error_logk=0.01, error_a0=0.01, error_a1=0.01, errordef=1.0, print_level=0)
fval1, params1 = attn_inst1.migrad()

logk_fit1 = params1[0]['value']
logk_fiterr1 = params1[0]['error']
a0_fit1 = params1[1]['value']
a0_fiterr1 = params1[1]['error']
a1_fit1 = params1[2]['value']
a1_fiterr1 = params1[2]['error']

#round2 w/ fcullen f_intr sed str
attn_inst2 = Minuit(chi_squared2, logk=logk_est, a0=a0_est, a1=a1_est, error_logk=0.01, error_a0=0.01, error_a1=0.01, errordef=1.0, print_level=0)
fval2, params2 = attn_inst2.migrad()

logk_fit2 = params2[0]['value']
logk_fiterr2 = params2[0]['error']
a0_fit2 = params2[1]['value']
a0_fiterr2 = params2[1]['error']
a1_fit2 = params2[2]['value']
a1_fiterr2 = params2[2]['error']

#round3 w/ fcullen f_intr sed neb
attn_inst3 = Minuit(chi_squared3, logk=logk_est, a0=a0_est, a1=a1_est, error_logk=0.01, error_a0=0.01, error_a1=0.01, errordef=1.0, print_level=0)
fval3, params3 = attn_inst3.migrad()

logk_fit3 = params3[0]['value']
logk_fiterr3 = params3[0]['error']
a0_fit3 = params3[1]['value']
a0_fiterr3 = params3[1]['error']
a1_fit3 = params3[2]['value']
a1_fiterr3 = params3[2]['error']

#A_lambda function, x=wavelength in microns
def A_lambda_function(x, a0, a1):
    return a0 * x**(-1.0) + a1 * x**(-2.0)

Av_1 = A_lambda_function(0.55, a0_fit1, a1_fit1) #1 from fintr power law
Av_2 = A_lambda_function(0.55, a0_fit2, a1_fit2) #2 from fc sed str
Av_3 = A_lambda_function(0.55, a0_fit3, a1_fit3) #3 from fc sed neb

# print(Av_1,Av_2, Av_3)
# print([logk_fit1,a0_fit1, a1_fit1])
# print([logk_fit2,a0_fit2, a1_fit2])
# print([logk_fit3,a0_fit3, a1_fit3])

attn_curve1 = A_lambda_function(cullen_wl*1.0E-4, a0_fit1, a1_fit1) / Av_1
attn_curve2 = A_lambda_function(cullen_wl*1.0E-4, a0_fit2, a1_fit2) / Av_2
attn_curve3 = A_lambda_function(cullen_wl*1.0E-4, a0_fit3, a1_fit3) / Av_3

#load in calzetti and smc curves for comparison
dustcurves = np.genfromtxt('dustcurves.csv', delimiter=',', skip_header=1, \
                            unpack=True)
smc_curve = dustcurves[1]
calz_curve = dustcurves[2]
wl_dustcurve = np.linspace(0.1,0.6,len(calz_curve))

###########################attn curve################################
# plt.figure()
#
# plt.plot(cullen_wl*1.0E-4, attn_curve1, color='tab:orange', label=r'intr $\beta$= '+str(beta_intrinsic))
# plt.plot(cullen_wl*1.0E-4, attn_curve2, color='firebrick', label='intr str SED')
# plt.plot(cullen_wl*1.0E-4, attn_curve3, color='fuchsia', label='intr neb SED')
#
# plt.plot(wl_dustcurve, smc_curve, label='SMC', linestyle=':', color='k')
# plt.plot(wl_dustcurve, calz_curve, label='Calzetti', linestyle='--', color='k')
#
# plt.xlabel(r'$\lambda$/$\mu$m')
# plt.ylabel(r'$A_{\lambda}/A_{V}$')
# plt.legend(fontsize='small')

####################contour plot from minuit##########################
# fig1 = plt.figure()
# ax1 = plt.subplot(221)
# plt.setp(ax1.get_xticklabels(), visible=False)
# attn_inst3.draw_mncontour('a0','logk')
# ax2 = plt.subplot(222)
# plt.setp(ax2.get_yticklabels(), visible=False)
# attn_inst3.draw_mncontour('a1','logk')
# ax3 = plt.subplot(223, sharex=ax1)
# attn_inst3.draw_mncontour('a0','a1')
# fig1.suptitle('Minuit Contours - fiby_bpass300bin_z4 Nebular SED (no H)')
# plt.show()

#########################################################################
#Artifical reddening test
Av_test = 1.0
#returns value of Alambda/Av for calzetti curve
Rv_calz = 4.05 #x below should be in um
Rv_smc = 2.74

def calzetti_dust(x):
    return ( 2.659*( -2.156 + 1.509*x**(-1.) - 0.198*x**(-2.) + 0.011*x**(-3.) ) +Rv_calz ) / Rv_calz

def smc_dust(x):

    c1 = -4.959
    c2 = 2.264
    c3 = 0.389
    c4 = 0.461
    x0 = 4.703
    gamma = 1.0

    def D_func(x,gamma,x0):
        return x**-2.0 / ( ( x**-2.0 - x0**-2.0 )**2.0 + (gamma**2.0)*(x**-2.0) )

    #note F=0 for 1/wl < 5.9, ie for wl > 0.169um, therefore this slightly off for 0.12<wl<0.169 >>>> will correct if needed
    def F_func(x):
        return 0.5392*( x**-1.0 - 5.9 )**2.0 + 0.05644*( x**-1.0 - 5.9 )**3

    smc_dust = (c1/Rv_smc + 1.0) + c2/(Rv_smc*x) + c3*D_func(x,gamma,x0)/Rv_smc + c4*F_func(x)/Rv_smc

    return smc_dust

calzetti_dust = calzetti_dust( fc_ia_wl*1.0E-4 )
smc_dust = smc_dust ( fc_ia_wl*1.0E-4 )

#can use f_intr from earlier section although could add some noise
#defining the artificially reddened sed as fo=fi*e^-(0.4*A_lambda) w/ 5% err
f_artificial_sed1 = f_intr * np.exp( -0.4 *  calzetti_dust * Av_test )
f_artificial_sed1_err = f_artificial_sed1 * 0.05
f_artificial_sed2 = f_intr * np.exp( -0.4 *  smc_dust * Av_test )
f_artificial_sed2_err = f_artificial_sed2 * 0.05

#making the ydata as usual for the fitting process.
y_artificial1 = 1.086*np.log(f_artificial_sed1/f_intr)
y_artificial1_err = 1.086*0.43429*(f_artificial_sed1_err/f_artificial_sed1)

y_artificial2 = 1.086*np.log(f_artificial_sed2/f_intr)
y_artificial2_err = 1.086*0.43429*(f_artificial_sed2_err/f_artificial_sed2)

def chi_squared_fake1(logk, a0, a1):
    return np.sum( ( ( y_artificial1 - model_curve(fc_ia_wl*1.0E-4, logk, a0, a1) ) / y_artificial1_err )**2.0 )

def chi_squared_fake2(logk, a0, a1):
    return np.sum( ( ( y_artificial2 - model_curve(fc_ia_wl*1.0E-4, logk, a0, a1) ) / y_artificial2_err )**2.0 )

#minuit minimisation for artificial 1, ie calzetti
attn_inst_fake1 = Minuit(chi_squared_fake1, logk=logk_est, a0=a0_est, a1=a1_est, error_logk=0.01, error_a0=0.01, error_a1=0.01, errordef=1.0, print_level=1)
fval_fake1, params_fake1 = attn_inst_fake1.migrad()

logk_fit_fake1 = params_fake1[0]['value']
logk_fiterr_fake1 = params_fake1[0]['error']
a0_fit_fake1 = params_fake1[1]['value']
a0_fiterr_fake1 = params_fake1[1]['error']
a1_fit_fake1 = params_fake1[2]['value']
a1_fiterr_fake1 = params_fake1[2]['error']

#minuit minimisation for artificial 2, ie smc
attn_inst_fake2 = Minuit(chi_squared_fake2, logk=logk_est, a0=a0_est, a1=a1_est, error_logk=0.01, error_a0=0.01, error_a1=0.01, errordef=1.0, print_level=1)
fval_fake2, params_fake2 = attn_inst_fake2.migrad()

logk_fit_fake2 = params_fake2[0]['value']
logk_fiterr_fake2 = params_fake2[0]['error']
a0_fit_fake2 = params_fake2[1]['value']
a0_fiterr_fake2 = params_fake2[1]['error']
a1_fit_fake2 = params_fake2[2]['value']
a1_fiterr_fake2 = params_fake2[2]['error']

Av_fake1 = A_lambda_function(0.55, a0_fit_fake1, a1_fit_fake1)
print(Av_fake1)
print([logk_fit_fake1,a0_fit_fake1, a1_fit_fake1])

Av_fake2 = A_lambda_function(0.55, a0_fit_fake2, a1_fit_fake2)
print(Av_fake2)
print([logk_fit_fake2,a0_fit_fake2, a1_fit_fake2])
