#this code marks beginning of v3 of attenuation curve fitting process

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from iminuit import Minuit
from pprint import pprint

#reading in catalogs
# subaru_photoz = pd.read_csv('ECDFS_subaru_zphot_IA738_neg_cut_JHK.csv', \
#                                         header=0, index_col=False, sep=',')
# subaru_specz = pd.read_csv('ECDFS_subaru_zspec_IA738_neg_cut_JHK.csv', \
#                                         header=0, index_col=False, sep=',')
clean_df = pd.read_csv('ryan_N453_clean_SFG_sample.csv', \
                                        header=0, index_col=False, sep=',')
#catalog in following form for above -
#num,ra,dec,b,v,r,i,j,h,k,ia427,ia445,ia484,ia505,ia527,ia550,ia574,ia598,
#ia624,ia651,ia679,ia709,ia739,ia767,ia797,ia856, ross cols  > each has F+E
flx_band_idx = np.array([i for i in range(3,len(clean_df.columns)-10,2)])

#print(clean_df.count()[flx_band_idx])

#band wavelengths used - note now inclu JHK and remove Ia464, IA827
band_wavelengths = np.array([.4511,.5396,.6517,.7838,1.2461,1.6534,2.1323, \
        .4273,.4456,.4842,.5063,.5242,.5512,.5743,.6000,.6226,.6502,.6788, \
        .7082,.7372,.7690,.7981,.8566])

#below is conversion factor for each band from fv to flambda
flux_space_conversion = 1.0E-32 * 2.998E8 * ( band_wavelengths * 1.0E-6 )**-2.0
#print('No. of objects in catalog - '+str(len(clean_df)))

#function takes in dataframe, and returns new df after sigma clipping
def sigma_clipping(df_input):
    df=df_input.copy()
    for i in range(1):

        for idx in flx_band_idx:

            condition = (df.iloc[:,idx] > df.iloc[:,idx].median() - 3.0*df.iloc[:,idx].std() ) & \
                            (df.iloc[:,idx] < df.iloc[:,idx].median() + 3.0*df.iloc[:,idx].std() )
            df.iloc[:,idx] = df.iloc[:,idx].loc[ condition ]

            condition2 = df.iloc[:,idx].isnull()
            df.iloc[:,idx+1] = df.iloc[:,idx+1].loc[~condition2]
    #return [flux_space_conversion*df.std()[flx_band_idx] / df.count()[flx_band_idx], flux_space_conversion*df.mean()[flx_band_idx]]
    return df

clipped_df = sigma_clipping(clean_df)
#print(clipped_df.count()[flx_band_idx] - clean_df.count()[flx_band_idx])

medianstack = clipped_df.median()[flx_band_idx] * flux_space_conversion
medianstack_err = 1.25 * clipped_df.std()[flx_band_idx] * flux_space_conversion \
                    / clipped_df.count()[flx_band_idx]

#medianstack_err = 0.05*medianstack
###############################################################################
#below here will be the fitting process#
#if in future doing normed bands then remove norm band from fit
##############################################################################

median_z = clipped_df['ZSPEC_NEW'].median()
band_wavelengths = band_wavelengths / (1. + median_z)

band_wavelengths_inter = band_wavelengths[7:]
medianstack_inter = medianstack[7:]
medianstack_err_inter = medianstack_err[7:]


def flux_curve(wavelength, norm, beta):
    return norm * wavelength**beta

def chi_squared(norm, beta):
    return np.sum( ( ( medianstack_inter - flux_curve(band_wavelengths_inter, norm, beta) \
                        ) / medianstack_err_inter )**2.0 )
# def chi_squared(norm, beta):
#     return np.sum( ( ( medianstack - flux_curve(band_wavelengths, norm, beta) \
#                         ) / medianstack_err )**2.0 )

beta_estimate = -1.5
#index of IA598 band
norm_estimate = medianstack[14] / band_wavelengths[14]**(beta_estimate)

min_inst = Minuit(chi_squared, norm=norm_estimate, beta=beta_estimate,         \
                    error_norm=0.1*norm_estimate, error_beta=0.1*beta_estimate,\
                    errordef=1, print_level=0)

chisq_min, results = min_inst.migrad()
# min_inst.hesse()
# min_inst.minos()
# pprint(min_inst.np_matrix())
# pprint(min_inst.np_matrix(correlation=True))


#fit results
normfit = results[0]['value']
normfit_err = results[0]['error']
betafit = results[1]['value']
betafit_err = results[1]['error']
# print(betafit,betafit_err)
# print(normfit,normfit_err)

#plotting of original data and fitted line
lambda_range = np.linspace(band_wavelengths.min(), band_wavelengths.max(),50)
flux_fitted = flux_curve(lambda_range,normfit,betafit)

###commented out to stop appearing when running subsequent codes.
# plt.figure()
# plt.errorbar(band_wavelengths,medianstack, yerr=medianstack*0.05, \
#              linestyle='None', marker='.', color='r', label='Photometry Median Stack')
# plt.errorbar(band_wavelengths_inter,medianstack_inter, linestyle='None', marker='+', color='tab:orange', label='Intermediate Bands')
# plt.plot(lambda_range,flux_fitted, color='k', label=r'$\beta_{fit}=$'+str(round(betafit,2)) )
# #plt.axvspan(band_wavelengths_inter.min(), band_wavelengths_inter.max(), facecolor='g', alpha=0.5)
# plt.xlabel(r'$\lambda$/$\mu$m')
# plt.ylabel(r'$F_{o}$ / $Wm^{-2}m_{-1}$')
# plt.legend(fontsize='small')
# #plt.xlim(xmax=0.3)
# #plt.ylim(ymax=3.75,ymin=2)
# #plt.savefig('medstack_betafit4_deshift.png')
# plt.show()
